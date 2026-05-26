import json
import unittest
from unittest.mock import patch

from agents import roles
from agents.roles import SmartMoneyAnalyst, build_smart_money_proxy_from_features, enforce_json_contract, parse_llm_json


class RolesJsonParsingTests(unittest.TestCase):
    def test_parse_llm_json_repairs_common_json_artifacts(self):
        parsed = parse_llm_json('```json\n{“sentiment”: “positive”, “confidence”: 0.71,}\n```')

        self.assertTrue(parsed["_parse_ok"])
        self.assertEqual(parsed["sentiment"], "positive")
        self.assertEqual(parsed["confidence"], 0.71)

    def test_parse_llm_json_fallback_extracts_signal_without_parse_phrase(self):
        parsed = parse_llm_json("判断：出货派发。理由：主力净流出明显。negative")

        self.assertFalse(parsed["_parse_ok"])
        self.assertEqual(parsed["sentiment"], "negative")
        self.assertNotIn("解析失败", parsed["thought_process"])

    def test_enforce_json_contract_adds_schema_to_plain_prompt(self):
        prompt = enforce_json_contract("请判断趋势。")

        self.assertIn("只输出一个合法 JSON 对象", prompt)
        self.assertIn('"sentiment"', prompt)
        self.assertIn('"confidence"', prompt)

    def test_smart_money_proxy_uses_existing_technical_features(self):
        proxy = build_smart_money_proxy_from_features(
            {
                "feature_values": {
                    "volume": {
                        "volume_ratio_5_20": 1.2,
                        "amount_ratio_5_20": 1.28,
                        "turnover": 3.63,
                        "turnover_percentile_60": 0.75,
                    },
                    "momentum": {"return_5d_pct": 1.54, "return_20d_pct": 19.63},
                    "trend": {"price_vs_ma20_pct": 7.67},
                }
            }
        )

        self.assertIn("主力资金代理", proxy)
        self.assertIn("近5/20日成交额比=1.28", proxy)
        self.assertIn("60日换手分位=0.75", proxy)

    def test_smart_money_uses_feature_proxy_when_provider_degrades(self):
        analyst = SmartMoneyAnalyst(name="主力资金子模块")
        features = {
            "feature_values": {
                "volume": {
                    "volume_ratio_5_20": 1.2,
                    "amount_ratio_5_20": 1.28,
                    "turnover": 3.63,
                    "turnover_percentile_60": 0.75,
                },
                "momentum": {"return_5d_pct": 1.54, "return_20d_pct": 19.63},
                "trend": {"price_vs_ma20_pct": 7.67},
            }
        }
        llm_payload = {
            "sentiment": "neutral",
            "confidence": 0.46,
            "reasoning": "资金活跃度尚可但缺少真实主力净流入，保持中性。",
            "thought_process": "使用量价资金代理。",
        }
        with (
            patch.object(roles.provider, "fetch_smart_money_data", return_value="[主力资金降级] 接口全部失败"),
            patch.object(analyst, "query_llm", return_value=json.dumps(llm_payload, ensure_ascii=False)),
        ):
            result = analyst.step("603155", target_date="2026-05-22", market_features=features)

        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["confidence"], 0.46)
        self.assertNotIn("_data_degraded", result)


if __name__ == "__main__":
    unittest.main()
