import unittest

import pandas as pd

from scripts.investment_advice import (
    _cache_matches_current_dl_backend,
    _compute_stability,
    _has_rule_fallback,
    _is_stable_cacheable_advice,
    _validate_advice_payload,
    assess_data_quality,
    build_marketless_direct_advice,
    build_multi_period_advice,
    build_technical_feature_bundle,
    calibrate_recommendation_by_quality,
)


class InvestmentAdviceQualityTests(unittest.TestCase):
    def test_high_quality_sources_score_well(self):
        quality = assess_data_quality(
            market_diagnostics={
                "source": "akshare",
                "fallback_used": False,
                "rows": 260,
                "as_of_date": "2026-05-20",
                "freshness_days": 0,
            },
            rag_diagnostics={
                "documents": 10,
                "fallback_used": False,
                "source_types": {"news": 6, "report": 4},
                "latest_date": "2026-05-18",
                "freshness_days": 2,
            },
            technical_case={"confidence": 0.72, "sentiment": "positive", "source_reports": []},
            fundamental_case={"confidence": 0.68, "sentiment": "positive", "source_reports": []},
        )

        self.assertEqual(quality["level"], "high")
        self.assertGreaterEqual(quality["score"], 0.75)
        self.assertIn("主要数据链路正常", quality["diagnostics"])

    def test_fallback_sources_lower_quality(self):
        quality = assess_data_quality(
            market_diagnostics={
                "source": "local_backtest_fallback",
                "fallback_used": True,
                "rows": 35,
                "as_of_date": "2025-01-01",
                "freshness_days": 300,
            },
            rag_diagnostics={
                "documents": 1,
                "fallback_used": True,
                "source_types": {"fallback": 1},
                "latest_date": "2025-01-01",
                "freshness_days": 300,
            },
            technical_case={
                "confidence": 0.25,
                "sentiment": "neutral",
                "source_reports": [{"confidence": 0.2, "sentiment": "neutral", "_parse_ok": False}],
            },
            fundamental_case={"confidence": 0.22, "sentiment": "neutral", "source_reports": []},
        )

        self.assertEqual(quality["level"], "low")
        self.assertLess(quality["score"], 0.45)
        self.assertIn("行情使用本地回测兜底", quality["diagnostics"])
        self.assertIn("新闻研报使用兜底文本", quality["diagnostics"])

    def test_marketless_direct_advice_is_valid_conservative_fallback(self):
        advice = build_marketless_direct_advice(
            "603157",
            {
                "source": "marketless_direct_fallback",
                "fallback_used": True,
                "rows": 0,
                "as_of_date": "2026-05-23",
                "freshness_days": None,
                "status": "market_unavailable",
                "errors": ["akshare_eastmoney_hist: timeout", "akshare_sina_daily: No value to decode"],
            },
        )

        self.assertEqual(_validate_advice_payload(advice), [])
        self.assertTrue(advice["marketless_fallback"])
        self.assertEqual(advice["recommendation"]["action"], "HOLD")
        self.assertEqual(advice["recommendation"]["position_percent"], 0.0)
        self.assertEqual(advice["data_quality"]["level"], "low")
        self.assertIn("行情数据不可用", advice["data_quality"]["diagnostics"][0])

    def test_low_quality_caps_directional_recommendation(self):
        risk = {"position_percent": 70.0, "reason": "原始风控理由"}
        recommendation = {
            "action": "BUY",
            "execution_action": "BUY 70.0%",
            "position_percent": 70.0,
            "confidence": 0.86,
        }

        calibrated = calibrate_recommendation_by_quality(
            recommendation,
            risk,
            {"score": 0.32, "note": "数据质量偏低。"},
        )

        self.assertEqual(calibrated["confidence"], 0.55)
        self.assertEqual(calibrated["position_percent"], 20.0)
        self.assertEqual(calibrated["execution_action"], "BUY 20.0%")
        self.assertTrue(risk["quality_adjusted"])
        self.assertIn("数据质量校准", risk["reason"])

    def test_hold_recommendation_keeps_position_when_quality_low(self):
        risk = {"position_percent": 0.0, "reason": "保持观望"}
        recommendation = {
            "action": "HOLD",
            "execution_action": "HOLD",
            "position_percent": 0.0,
            "confidence": 0.7,
        }

        calibrated = calibrate_recommendation_by_quality(
            recommendation,
            risk,
            {"score": 0.2, "note": "数据质量偏低。"},
        )

        self.assertEqual(calibrated["confidence"], 0.7)
        self.assertEqual(calibrated["position_percent"], 0.0)
        self.assertFalse(risk.get("quality_adjusted", False))

    def test_directional_zero_position_calibrates_to_hold(self):
        risk = {"decision": "BUY", "position_percent": 0.0, "reason": "风控未给仓位"}
        recommendation = {
            "action": "BUY",
            "execution_action": "BUY",
            "position_percent": 0.0,
            "confidence": 0.8,
        }

        calibrated = calibrate_recommendation_by_quality(
            recommendation,
            risk,
            {"score": 0.32, "note": "数据质量偏低。"},
        )

        self.assertEqual(calibrated["action"], "HOLD")
        self.assertEqual(calibrated["execution_action"], "HOLD")
        self.assertEqual(calibrated["position_percent"], 0.0)
        self.assertEqual(risk["decision"], "HOLD")
        self.assertIn("方向性仓位为0", risk["reason"])

    def test_multi_period_advice_returns_three_decision_views(self):
        technical_features = {
            "trend": {
                "close": 10.0,
                "ma20": 9.6,
                "price_vs_ma20_pct": 4.17,
                "ma20_slope_5d_pct": 1.2,
            },
            "momentum": {
                "return_5d_pct": 2.4,
                "return_20d_pct": 8.2,
                "rsi14": 61.0,
                "macd_hist": 0.08,
            },
            "risk": {"atr14_pct": 3.0},
            "volume": {"volume_ratio_5_20": 1.45},
            "signals": [
                "价格位于 MA20 上方，但均线结构尚未完全顺排。",
                "MACD 柱体为正，动量偏多。",
                "近5日成交量显著高于20日均量，资金活跃度提升。",
            ],
        }

        multi = build_multi_period_advice(
            recommendation={"action": "BUY", "confidence": 0.72, "position_percent": 30.0},
            risk={"decision": "BUY", "position_percent": 30.0, "reason": "风控通过"},
            referee={"decision": "BUY", "confidence": 0.72, "trend_strength": 0.45},
            analyst_cases={
                "technical_flow": {"sentiment": "positive", "confidence": 0.7},
                "fundamental_news": {"sentiment": "positive", "confidence": 0.68},
            },
            technical_features=technical_features,
            data_quality={"score": 0.82},
        )

        self.assertEqual(set(multi.keys()), {"short_term", "swing_term", "risk_plan"})
        self.assertEqual(multi["short_term"]["horizon"], "1-3 trading days")
        self.assertIn(multi["short_term"]["action"], {"BUY", "SELL", "HOLD"})
        self.assertEqual(multi["swing_term"]["horizon"], "2-4 weeks")
        self.assertGreaterEqual(multi["short_term"]["confidence"], 0.35)
        self.assertIn("stop_loss", multi["risk_plan"])
        self.assertGreaterEqual(len(multi["risk_plan"]["invalid_conditions"]), 2)

    def test_multi_period_sell_reason_does_not_reuse_bullish_macd_signal(self):
        multi = build_multi_period_advice(
            recommendation={"action": "HOLD", "confidence": 0.81, "position_percent": 0.0},
            risk={"decision": "HOLD", "position_percent": 0.0, "reason": "风控观望"},
            referee={"decision": "HOLD", "confidence": 0.81, "trend_strength": 0.2},
            analyst_cases={
                "technical_flow": {"sentiment": "negative", "confidence": 0.65},
                "fundamental_news": {"sentiment": "negative", "confidence": 0.85},
            },
            technical_features={
                "trend": {"close": 44.78, "ma20": 41.49, "price_vs_ma20_pct": 7.94, "ma20_slope_5d_pct": 1.0},
                "momentum": {"return_5d_pct": -4.28, "return_20d_pct": 34.39, "rsi14": 66.28, "macd_hist": 0.08},
                "risk": {"atr14_pct": 7.87, "max_drawdown60_pct": -16.48},
                "volume": {"volume_ratio_5_20": 1.70},
                "signals": [
                    "价格位于 MA20 上方，但均线结构尚未完全顺排。",
                    "MACD 柱体为正，动量偏多。",
                    "ATR14 占比较高，短线波动风险偏大。",
                    "近60日最大回撤较深，趋势修复需要更多确认。",
                ],
            },
            data_quality={"score": 0.86},
        )

        self.assertEqual(multi["short_term"]["action"], "SELL")
        self.assertNotEqual(multi["short_term"]["reason"], "MACD 柱体为正，动量偏多。")
        self.assertIn("风险", multi["short_term"]["reason"])

    def test_stability_counts_structured_parse_failures(self):
        advice = {
            "recommendation": {"action": "HOLD", "confidence": 0.7, "reason": "保持观望"},
            "risk": {"decision": "HOLD", "reason": "风控观望"},
            "referee": {"decision": "HOLD", "confidence": 0.7},
            "analyst_cases": {
                "technical_flow": {
                    "sentiment": "negative",
                    "confidence": 0.6,
                    "reasoning": "技术承压",
                    "source_reports": [
                        {
                            "agent": "技术面子模块",
                            "sentiment": "negative",
                            "confidence": 0.5,
                            "reasoning": "不是靠文本解析失败四个字统计",
                            "_parse_ok": False,
                            "_parse_error": "unit test",
                        }
                    ],
                },
                "fundamental_news": {
                    "sentiment": "negative",
                    "confidence": 0.6,
                    "reasoning": "估值承压",
                },
            },
        }

        stability = _compute_stability(advice)

        self.assertEqual(stability["stability_diagnostics"]["parse_fail_count"], 1)
        self.assertEqual(
            stability["stability_diagnostics"]["parse_failures"][0]["agent"],
            "技术面子模块",
        )

    def test_rule_fallback_advice_is_not_stable_cacheable(self):
        advice = {
            "recommendation": {"action": "HOLD", "reason": "保持观望"},
            "risk": {"decision": "HOLD", "reason": "风控观望"},
            "referee": {"decision": "HOLD"},
            "analyst_cases": {
                "technical_flow": {
                    "reasoning": "[规则降级] 外部LLM暂不可用，改用关键词规则推断；当前倾向: neutral。",
                },
                "fundamental_news": {"reasoning": "估值中性"},
            },
            "stability_diagnostics": {"rule_fallback_count": 1},
        }

        self.assertTrue(_has_rule_fallback(advice))
        self.assertFalse(_is_stable_cacheable_advice(advice))

    def test_llm_error_text_advice_is_not_stable_cacheable(self):
        advice = {
            "recommendation": {"action": "HOLD", "reason": "保持观望"},
            "risk": {"decision": "HOLD", "reason": "触发原因: url error: CERTIFICATE_VERIFY_FAILED"},
            "referee": {"decision": "HOLD"},
            "analyst_cases": {
                "technical_flow": {"reasoning": "技术中性"},
                "fundamental_news": {"reasoning": "估值中性"},
            },
            "stability_diagnostics": {"rule_fallback_count": 0},
        }

        self.assertTrue(_has_rule_fallback(advice))
        self.assertFalse(_is_stable_cacheable_advice(advice))

    def test_normal_advice_is_stable_cacheable(self):
        advice = {
            "recommendation": {"action": "HOLD", "reason": "量价分歧，暂时观望"},
            "risk": {"decision": "HOLD", "reason": "仓位控制"},
            "referee": {"decision": "HOLD"},
            "analyst_cases": {
                "technical_flow": {"reasoning": "资金面中性"},
                "fundamental_news": {"reasoning": "新闻面中性"},
            },
            "stability_diagnostics": {"rule_fallback_count": 0},
        }

        self.assertFalse(_has_rule_fallback(advice))
        self.assertTrue(_is_stable_cacheable_advice(advice))

    def test_kronos_backend_rejects_old_lstm_stable_cache(self):
        old_advice = {
            "analyst_cases": {
                "technical_flow": {
                    "source_reports": [
                        {"agent": "DL量化子模块", "model": "lstm", "reasoning": "LSTM 输出"}
                    ]
                }
            }
        }
        kronos_advice = {
            "analyst_cases": {
                "technical_flow": {
                    "source_reports": [
                        {"agent": "DL量化子模块", "model": "kronos", "prediction": {"model": "kronos"}}
                    ]
                }
            }
        }

        import os

        previous = os.environ.get("DL_BACKEND")
        try:
            os.environ["DL_BACKEND"] = "auto"
            self.assertFalse(_cache_matches_current_dl_backend(old_advice))
            self.assertTrue(_cache_matches_current_dl_backend(kronos_advice))
        finally:
            if previous is None:
                os.environ.pop("DL_BACKEND", None)
            else:
                os.environ["DL_BACKEND"] = previous

    def test_technical_feature_bundle_keeps_dl_matrix_and_adds_indicators(self):
        rows = []
        for idx in range(80):
            close = 10.0 + idx * 0.08
            open_px = close - 0.03
            prev_close = 10.0 + max(idx - 1, 0) * 0.08
            rows.append(
                {
                    "日期": f"2026-03-{(idx % 28) + 1:02d}",
                    "开盘": open_px,
                    "收盘": close,
                    "最高": close + 0.12,
                    "最低": open_px - 0.08,
                    "成交量": 1_000_000 + idx * 10_000,
                    "成交额": (1_000_000 + idx * 10_000) * close,
                    "振幅": 2.0,
                    "涨跌幅": (close - prev_close) / prev_close * 100.0,
                    "涨跌额": close - prev_close,
                    "换手率": 1.0 + idx * 0.01,
                }
            )
        bundle = build_technical_feature_bundle(pd.DataFrame(rows))

        self.assertEqual(bundle["dl_features"].shape, (10, 10))
        self.assertEqual(len(bundle["kronos_ohlcv"]), 80)
        self.assertEqual(
            set(bundle["kronos_ohlcv"][0].keys()),
            {"timestamps", "open", "high", "low", "close", "volume", "amount"},
        )
        self.assertIn("趋势", bundle["summary_text"])
        features = bundle["feature_values"]
        self.assertEqual(features["version"], "technical_features_v1")
        self.assertEqual(features["kronos_ohlcv"]["rows"], 80)
        self.assertEqual(features["kronos_ohlcv"]["max_context"], 512)
        self.assertIn("ma20", features["trend"])
        self.assertIn("rsi14", features["momentum"])
        self.assertGreater(features["momentum"]["rsi14"], 70)
        self.assertIn("atr14_pct", features["risk"])
        self.assertGreater(len(features["signals"]), 0)


if __name__ == "__main__":
    unittest.main()
