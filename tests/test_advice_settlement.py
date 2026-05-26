import os
import tempfile
import unittest
from unittest.mock import patch

from backend import services
from backend import strategy_rules


class AdviceSettlementTests(unittest.TestCase):
    def _write_advice(self, advice_dir: str, name: str = "000001_20240101_090000.json") -> str:
        path = os.path.join(advice_dir, name)
        services.write_json(
            path,
            {
                "ticker": "000001",
                "as_of_date": "2024-01-01",
                "recommendation": {
                    "action": "BUY",
                    "confidence": 0.76,
                    "position_percent": 100,
                    "reason": "测试建议",
                },
                "referee": {"decision": "BUY", "confidence": 0.72},
                "analyst_cases": {
                    "technical_flow": {"sentiment": "positive", "confidence": 0.8},
                    "fundamental_news": {"sentiment": "negative", "confidence": 0.7},
                },
            },
        )
        return path

    def _price_map(self) -> dict[str, float]:
        prices = {"2024-01-01": 10.0}
        for idx in range(1, 23):
            prices[f"2024-01-{idx + 1:02d}"] = 10.0 + idx
        return prices

    def test_settlement_creates_one_record_per_horizon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            advice_dir = os.path.join(tmpdir, "advice")
            os.makedirs(advice_dir)
            state_path = os.path.join(tmpdir, "advice_settlements.json")
            reflections_path = os.path.join(tmpdir, "reflections.json")
            self._write_advice(advice_dir)

            with (
                patch.object(services, "ADVICE_DIR", advice_dir),
                patch.object(services, "ADVICE_SETTLEMENT_PATH", state_path),
                patch.object(services, "REFLECTIONS_PATH", reflections_path),
                patch.object(strategy_rules, "STRATEGY_RULES_PATH", os.path.join(tmpdir, "strategy_rules.json")),
                patch.object(services, "_load_price_map_for_ticker", return_value=self._price_map()),
                patch.object(services, "build_dashboard_summary", return_value={"updated_at": "now"}),
            ):
                result = services.settle_advice_experience(base_dir=tmpdir, max_items=100)

            self.assertEqual(result["settled_count"], 5)
            self.assertEqual([x["horizon"] for x in result["evaluation"]["horizons"]], ["T+1", "T+3", "T+5", "T+10", "T+20"])

            reflections = services.read_json(reflections_path, [])
            self.assertEqual(len(reflections), 5)
            self.assertEqual({r["horizon"] for r in reflections}, {"T+1", "T+3", "T+5", "T+10", "T+20"})
            self.assertTrue(all(r["confidence_bucket"] == "0.7-0.8" for r in reflections))
            self.assertTrue(all("signal_profile" in r for r in reflections))
            self.assertTrue(all("baseline_returns" in r for r in reflections))
            self.assertTrue(all("mistake_attribution" in r for r in reflections))
            self.assertIn("buy_hold", reflections[0]["baseline_returns"])
            self.assertIn("excess_return_vs_best_baseline", reflections[0])

    def test_legacy_file_key_only_skips_t1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            advice_dir = os.path.join(tmpdir, "advice")
            os.makedirs(advice_dir)
            state_path = os.path.join(tmpdir, "advice_settlements.json")
            reflections_path = os.path.join(tmpdir, "reflections.json")
            advice_name = "000001_20240101_090000.json"
            self._write_advice(advice_dir, advice_name)
            services.write_json(state_path, {"settled_keys": [advice_name], "records": []})

            with (
                patch.object(services, "ADVICE_DIR", advice_dir),
                patch.object(services, "ADVICE_SETTLEMENT_PATH", state_path),
                patch.object(services, "REFLECTIONS_PATH", reflections_path),
                patch.object(strategy_rules, "STRATEGY_RULES_PATH", os.path.join(tmpdir, "strategy_rules.json")),
                patch.object(services, "_load_price_map_for_ticker", return_value=self._price_map()),
                patch.object(services, "build_dashboard_summary", return_value={"updated_at": "now"}),
            ):
                result = services.settle_advice_experience(base_dir=tmpdir, max_items=100)

            self.assertEqual(result["settled_count"], 4)
            self.assertEqual({r["horizon"] for r in services.read_json(reflections_path, [])}, {"T+3", "T+5", "T+10", "T+20"})

    def test_settlement_can_target_one_horizon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            advice_dir = os.path.join(tmpdir, "advice")
            os.makedirs(advice_dir)
            state_path = os.path.join(tmpdir, "advice_settlements.json")
            reflections_path = os.path.join(tmpdir, "reflections.json")
            self._write_advice(advice_dir)

            with (
                patch.object(services, "ADVICE_DIR", advice_dir),
                patch.object(services, "ADVICE_SETTLEMENT_PATH", state_path),
                patch.object(services, "REFLECTIONS_PATH", reflections_path),
                patch.object(strategy_rules, "STRATEGY_RULES_PATH", os.path.join(tmpdir, "strategy_rules.json")),
                patch.object(services, "_load_price_map_for_ticker", return_value=self._price_map()),
                patch.object(services, "build_dashboard_summary", return_value={"updated_at": "now"}),
            ):
                result = services.settle_advice_experience(base_dir=tmpdir, max_items=100, horizons=[5])

            self.assertEqual(result["settled_count"], 1)
            self.assertEqual(result["horizons"], ["T+5"])
            reflections = services.read_json(reflections_path, [])
            self.assertEqual(len(reflections), 1)
            self.assertEqual(reflections[0]["horizon"], "T+5")

    def test_evaluation_calibrates_confidence_and_agent_reliability(self):
        records = [
            {
                "source": "advice_settlement",
                "ticker": "000001",
                "decision": "BUY",
                "confidence": 0.76,
                "confidence_bucket": "0.7-0.8",
                "horizon": "T+5",
                "pnl_percent": 2.0,
                "market_move_percent": 2.0,
                "as_of_date": "2024-01-01",
                "settled_date": "2024-01-08",
                "signal_profile": {
                    "stock_category": "深市主板",
                    "agents": {
                        "technical_flow": {"action": "BUY", "confidence": 0.8},
                        "fundamental_news": {"action": "SELL", "confidence": 0.7},
                    },
                    "conflict": {
                        "technical_vs_fundamental": True,
                        "technical_action": "BUY",
                        "fundamental_action": "SELL",
                    },
                },
            },
            {
                "source": "advice_settlement",
                "ticker": "000002",
                "decision": "BUY",
                "confidence": 0.74,
                "confidence_bucket": "0.7-0.8",
                "horizon": "T+5",
                "pnl_percent": -1.0,
                "market_move_percent": -1.0,
                "as_of_date": "2024-01-02",
                "settled_date": "2024-01-09",
                "signal_profile": {
                    "stock_category": "深市主板",
                    "agents": {
                        "technical_flow": {"action": "SELL", "confidence": 0.6},
                        "fundamental_news": {"action": "BUY", "confidence": 0.7},
                    },
                    "conflict": {
                        "technical_vs_fundamental": True,
                        "technical_action": "SELL",
                        "fundamental_action": "BUY",
                    },
                },
            },
        ]

        evaluation = services.build_advice_settlement_evaluation(records)

        self.assertEqual(evaluation["horizons"][0]["horizon"], "T+5")
        self.assertEqual(evaluation["horizons"][0]["hit_rate"], 0.5)
        self.assertEqual(evaluation["confidence_calibration"][0]["confidence_bucket"], "0.7-0.8")
        self.assertEqual(evaluation["confidence_calibration"][0]["hit_rate"], 0.5)
        tech = next(x for x in evaluation["agent_reliability"] if x["agent"] == "technical_flow")
        fund = next(x for x in evaluation["agent_reliability"] if x["agent"] == "fundamental_news")
        self.assertEqual(tech["hit_rate"], 1.0)
        self.assertEqual(fund["hit_rate"], 0.0)
        self.assertEqual(evaluation["conflict_metrics"][0]["better_side"], "technical")

    def test_quality_ranking_aggregates_baselines_and_mistakes(self):
        records = [
            {
                "source": "advice_settlement",
                "ticker": "000001",
                "decision": "BUY",
                "confidence": 0.8,
                "horizon": "T+5",
                "pnl_percent": 2.0,
                "excess_return_vs_buy_hold": 0.0,
                "excess_return_vs_best_baseline": -0.5,
                "reward_score": 0.5,
                "mistake_attribution": {"mistake_type": "no_mistake"},
            },
            {
                "source": "advice_settlement",
                "ticker": "000002",
                "decision": "BUY",
                "confidence": 0.82,
                "horizon": "T+5",
                "pnl_percent": -3.0,
                "excess_return_vs_buy_hold": -6.0,
                "excess_return_vs_best_baseline": -6.0,
                "reward_score": -0.6,
                "mistake_attribution": {"mistake_type": "false_breakout", "root_cause": "unit"},
            },
        ]
        with patch.object(services, "_load_settlement_records_for_analytics", return_value=records):
            result = services.build_advice_quality_ranking(limit=10)

        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["ranking"][0]["ticker"], "000001")
        self.assertIn("false_breakout", result["mistake_distribution"])
        self.assertEqual(result["worst_cases"][0]["ticker"], "000002")

    def test_quality_ranking_infers_legacy_missing_mistake_attribution(self):
        records = [
            {
                "source": "advice_settlement",
                "ticker": "000001",
                "decision": "BUY",
                "confidence": 0.8,
                "horizon": "T+5",
                "pnl_percent": -2.0,
                "market_move_percent": -1.4,
                "excess_return_vs_best_baseline": -3.0,
                "reward_score": -0.5,
            },
            {
                "source": "advice_settlement",
                "ticker": "000001",
                "decision": "BUY",
                "confidence": 0.7,
                "horizon": "T+3",
                "pnl_percent": 1.0,
                "market_move_percent": 0.5,
                "excess_return_vs_best_baseline": 0.0,
                "reward_score": 0.2,
            },
        ]
        with patch.object(services, "_load_settlement_records_for_analytics", return_value=records):
            result = services.build_advice_quality_ranking(limit=10)

        self.assertEqual(result["inferred_attribution_count"], 2)
        self.assertIn("false_breakout", result["mistake_distribution"])
        self.assertIn("no_mistake", result["mistake_distribution"])
        self.assertEqual(result["ranking"][0]["dominant_mistake_type"], "false_breakout")
        self.assertEqual(result["worst_cases"][0]["mistake_attribution"]["mistake_type"], "false_breakout")
        self.assertTrue(result["worst_cases"][0]["mistake_attribution"]["inferred_for_analytics"])

    def test_counterfactual_simulation_changes_action_on_stop_loss(self):
        with tempfile.TemporaryDirectory() as advice_dir:
            self._write_advice(advice_dir)
            with patch.object(services, "ADVICE_DIR", advice_dir):
                result = services.simulate_advice_counterfactual(
                    ticker="000001",
                    scenario={
                        "price_change_pct": -4,
                        "market_index_change_pct": -2,
                        "user_position_percent": 30,
                        "breaks_stop_loss": True,
                    },
                    user_profile={"risk_profile": "conservative", "current_position_percent": 30},
                )

        self.assertEqual(result["original_action"], "BUY")
        self.assertEqual(result["simulated_action"], "SELL")
        self.assertTrue(result["changed"])

    def test_mistake_attribution_updates_strategy_rule_library(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rule_path = os.path.join(tmpdir, "strategy_rules.json")
            record = {
                "source": "advice_settlement",
                "ticker": "000001",
                "decision": "BUY",
                "horizon": "T+5",
                "pnl_percent": -3.0,
                "excess_return_vs_best_baseline": -5.0,
                "settled_date": "2024-01-08",
                "settled_at": "2024-01-08 15:30:00",
                "signal_profile": {"stock_category": "深市主板"},
                "mistake_attribution": {
                    "mistake_type": "false_breakout",
                    "severity": "high",
                    "root_cause": "买入后价格未延续。",
                    "future_rule": "突破类买入需增加次日确认、量能延续或回踩不破条件。",
                    "tags": ["买入后价格反向"],
                },
            }
            with patch.object(strategy_rules, "STRATEGY_RULES_PATH", rule_path):
                result = strategy_rules.update_strategy_rules_from_records([record])
                payload = strategy_rules.load_strategy_rules()

            self.assertEqual(result["inserted"], 1)
            self.assertEqual(payload["summary"]["active_rules"], 1)
            self.assertEqual(payload["rules"][0]["mistake_type"], "false_breakout")
            self.assertIn("突破类买入", payload["rules"][0]["rule_text"])

    def test_strategy_rules_display_groups_duplicate_rule_text(self):
        rules = [
            {
                "id": "sr_a",
                "status": "active",
                "mistake_type": "false_breakout",
                "action": "BUY",
                "horizon": "T+5",
                "stock_category": "沪市主板",
                "rule_text": "突破类买入需增加次日确认、量能延续或回踩不破条件。",
                "support_count": 6,
                "severity_score": 13,
                "avg_pnl_percent": -1.08,
                "avg_excess_vs_best_baseline": -5.36,
                "source_examples": [{"ticker": "603156", "horizon": "T+5", "pnl_percent": -0.64}],
            },
            {
                "id": "sr_b",
                "status": "active",
                "mistake_type": "false_breakout",
                "action": "BUY",
                "horizon": "T+3",
                "stock_category": "沪市主板",
                "rule_text": "突破类买入需增加次日确认、量能延续或回踩不破条件。",
                "support_count": 3,
                "severity_score": 8,
                "avg_pnl_percent": -1.44,
                "avg_excess_vs_best_baseline": -11.22,
                "source_examples": [{"ticker": "600519", "horizon": "T+3", "pnl_percent": -0.07}],
            },
        ]

        grouped = strategy_rules.group_strategy_rules_for_display(rules)

        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["support_count"], 9)
        self.assertEqual(grouped[0]["severity_score"], 21)
        self.assertEqual(grouped[0]["source_rule_count"], 2)
        self.assertIn("T+5", grouped[0]["horizons"])
        self.assertIn("T+3", grouped[0]["horizons"])

    def test_strategy_rules_adjust_future_buy_advice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rule_path = os.path.join(tmpdir, "strategy_rules.json")
            services.write_json(
                rule_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "sr_unit",
                            "status": "active",
                            "mistake_type": "false_breakout",
                            "action": "BUY",
                            "stock_category": "深市主板",
                            "rule_text": "突破类买入需增加次日确认。",
                            "support_count": 2,
                            "severity_score": 4,
                            "effect": {"confidence_delta": -0.05, "position_multiplier": 0.75},
                        }
                    ],
                    "summary": {"active_rules": 1},
                },
            )
            advice = {
                "ticker": "000001",
                "recommendation": {"action": "BUY", "confidence": 0.8, "position_percent": 40.0, "reason": "测试买入"},
                "risk": {"decision": "BUY", "position_percent": 40.0, "reason": "测试风控"},
                "analyst_cases": {},
                "data_quality": {"score": 0.8},
                "referee": {"trend_strength": 0.7},
            }
            with patch.object(strategy_rules, "STRATEGY_RULES_PATH", rule_path):
                adjusted = strategy_rules.apply_strategy_rules_to_advice(advice)

            self.assertTrue(adjusted["strategy_rules"]["applied"])
            self.assertEqual(adjusted["recommendation"]["position_percent"], 30.0)
            self.assertLess(adjusted["recommendation"]["confidence"], 0.8)
            self.assertIn("策略规则库校准", adjusted["recommendation"]["reason"])

    def test_strategy_rules_zero_position_becomes_hold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rule_path = os.path.join(tmpdir, "strategy_rules.json")
            services.write_json(
                rule_path,
                {
                    "version": 1,
                    "rules": [
                        {
                            "id": "sr_zero",
                            "status": "active",
                            "mistake_type": "false_breakout",
                            "action": "BUY",
                            "stock_category": "深市主板",
                            "rule_text": "突破类买入需增加次日确认。",
                            "support_count": 2,
                            "severity_score": 4,
                            "effect": {"confidence_delta": -0.05, "position_multiplier": 0.75},
                        }
                    ],
                    "summary": {"active_rules": 1},
                },
            )
            advice = {
                "ticker": "000001",
                "recommendation": {"action": "BUY", "confidence": 0.8, "position_percent": 0.0, "reason": "测试买入"},
                "risk": {"decision": "BUY", "position_percent": 0.0, "reason": "测试风控"},
                "analyst_cases": {},
                "data_quality": {"score": 0.8},
                "referee": {"trend_strength": 0.7},
            }

            with patch.object(strategy_rules, "STRATEGY_RULES_PATH", rule_path):
                adjusted = strategy_rules.apply_strategy_rules_to_advice(advice)

            self.assertEqual(adjusted["recommendation"]["action"], "HOLD")
            self.assertEqual(adjusted["recommendation"]["execution_action"], "HOLD")
            self.assertEqual(adjusted["risk"]["decision"], "HOLD")
            self.assertIn("方向性仓位为0", adjusted["recommendation"]["reason"])

    def test_portfolio_advice_uses_latest_single_stock_advice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            advice_dir = os.path.join(tmpdir, "advice")
            os.makedirs(advice_dir)
            self._write_advice(advice_dir)
            with (
                patch.object(services, "ADVICE_DIR", advice_dir),
                patch.object(services, "load_strategy_rules", return_value={"summary": {"active_rules": 0}, "rules": []}),
                patch.object(services, "apply_strategy_rules_to_advice", side_effect=lambda advice: advice),
            ):
                result = services.build_portfolio_advice(
                    holdings=[{"ticker": "000001", "weight_percent": 5, "cost_price": 9.5}],
                    user_profile={"risk_profile": "balanced", "max_position_per_stock": 20},
                    cash_percent=10,
                    objective="balanced",
                )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["holdings"][0]["ticker"], "000001")
            self.assertEqual(result["holdings"][0]["action"], "BUY")
            self.assertGreaterEqual(result["holdings"][0]["target_weight_percent"], 5.0)


if __name__ == "__main__":
    unittest.main()
