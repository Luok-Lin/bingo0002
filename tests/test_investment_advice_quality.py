import unittest

from scripts.investment_advice import (
    _compute_stability,
    assess_data_quality,
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


if __name__ == "__main__":
    unittest.main()
