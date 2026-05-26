import json
import unittest
from unittest.mock import patch

from agents.llm_client import LLMClientConfig
from backend import report_narrative


class ReportNarrativeTests(unittest.TestCase):
    def _advice(self):
        return {
            "ticker": "600519",
            "generated_at": "2026-05-23 10:00:00",
            "as_of_date": "2026-05-23",
            "latest_market": {"close": 10.5, "pct_change": 1.2, "turnover": 2.1},
            "recommendation": {
                "action": "HOLD",
                "execution_action": "HOLD",
                "position_percent": 20.0,
                "confidence": 0.72,
                "reason": "测试建议理由：量价仍需确认。",
            },
            "risk": {"decision": "HOLD", "position_percent": 20.0, "reason": "跌破关键支撑则减仓。"},
            "data_quality": {
                "score": 0.8,
                "level": "high",
                "note": "测试数据质量良好。",
                "rag": {"documents": 3, "source_types": {"news": 2, "report": 1}},
            },
            "stability_score": 0.82,
            "stability_level": "medium",
            "stability_note": "建议较稳定。",
            "stability_diagnostics": {
                "parse_fail_count": 0,
                "rule_fallback_count": 1,
                "empty_reason_count": 0,
                "consensus_tech_fund": True,
                "referee_action": "SELL",
                "risk_action": "HOLD",
                "recommendation_action": "HOLD",
            },
        }

    def test_deterministic_narrative_hides_internal_field_names(self):
        narrative = report_narrative.build_report_narrative(self._advice(), use_llm=False)
        raw = json.dumps(narrative, ensure_ascii=False)

        self.assertIn("执行", narrative["executive_summary"])
        for token in report_narrative.FORBIDDEN_FIELD_TOKENS:
            self.assertNotIn(token, raw)

    def test_llm_narrative_is_used_when_valid(self):
        payload = {
            "executive_summary": "LLM执行摘要。",
            "investment_thesis": "LLM核心逻辑。",
            "evidence_review": "LLM证据链。",
            "risk_review": "LLM风险。",
            "stability_review": "LLM稳定度。",
            "execution_plan": "LLM执行计划。",
            "watchlist": ["观察量能", "跟踪公告"],
        }

        fake_config = LLMClientConfig(
            api_key="configured",
            base_url="https://example.invalid/v1",
            model_name="test",
            stable_mode=True,
            json_retry=1,
            timeout_seconds=5,
            temperature=0.1,
            top_p=0.2,
        )
        with (
            patch.object(report_narrative.LLMClient, "from_env", return_value=fake_config),
            patch.object(report_narrative.LLMClient, "query", return_value=json.dumps(payload, ensure_ascii=False)),
        ):
            narrative = report_narrative.build_report_narrative(self._advice(), use_llm=True)

        self.assertEqual(narrative["executive_summary"], "LLM执行摘要。")
        self.assertEqual(narrative["_source"], "llm")

    def test_default_uses_llm_when_available(self):
        payload = {
            "executive_summary": "默认LLM执行摘要。",
            "investment_thesis": "默认LLM核心逻辑。",
            "evidence_review": "默认LLM证据链。",
            "risk_review": "默认LLM风险。",
            "stability_review": "默认LLM稳定度。",
            "execution_plan": "默认LLM执行计划。",
            "watchlist": ["观察量能"],
        }
        fake_config = LLMClientConfig(
            api_key="configured",
            base_url="https://example.invalid/v1",
            model_name="test",
            stable_mode=True,
            json_retry=1,
            timeout_seconds=5,
            temperature=0.1,
            top_p=0.2,
        )
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(report_narrative.LLMClient, "from_env", return_value=fake_config),
            patch.object(report_narrative.LLMClient, "query", return_value=json.dumps(payload, ensure_ascii=False)),
        ):
            narrative = report_narrative.build_report_narrative(self._advice())

        self.assertEqual(narrative["executive_summary"], "默认LLM执行摘要。")
        self.assertEqual(narrative["_source"], "llm")


if __name__ == "__main__":
    unittest.main()
