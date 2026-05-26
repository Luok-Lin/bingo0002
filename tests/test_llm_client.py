import json
import unittest

from agents.llm_client import (
    LLMClient,
    LLMClientConfig,
    extract_json_object,
    is_valid_json_payload,
    repair_json_text,
    rule_based_fallback,
)


class LLMClientTests(unittest.TestCase):
    def test_extract_json_object_from_markdown(self):
        raw = "```json\n{\"sentiment\":\"neutral\", \"confidence\":0.5}\n```"
        self.assertEqual(extract_json_object(raw), "{\"sentiment\":\"neutral\", \"confidence\":0.5}")

    def test_schema_validation_requires_minimum_keys(self):
        self.assertTrue(is_valid_json_payload('{"sentiment":"neutral","confidence":0.5}', ["sentiment", "confidence"]))
        self.assertFalse(is_valid_json_payload('{"sentiment":"neutral"}', ["sentiment", "confidence", "reasoning"]))

    def test_repair_json_text_handles_common_model_artifacts(self):
        raw = '结论如下：```json\n{“sentiment”: “negative”, “confidence”: 0.62,}\n```'
        self.assertEqual(repair_json_text(raw), '{"sentiment": "negative", "confidence": 0.62}')
        self.assertTrue(is_valid_json_payload(raw, ["sentiment", "confidence"]))

    def test_rule_fallback_is_json(self):
        payload = json.loads(rule_based_fallback("上涨 突破 利好", "unit test"))
        self.assertIn(payload["sentiment"], {"positive", "negative", "neutral"})
        self.assertIn("confidence", payload)

    def test_unconfigured_key_uses_fallback(self):
        cfg = LLMClientConfig(
            api_key="your_api_key_here",
            base_url="https://example.invalid/v1",
            model_name="test",
            stable_mode=True,
            json_retry=1,
            timeout_seconds=5,
            temperature=0.1,
            top_p=0.2,
        )
        payload = json.loads(LLMClient(cfg).query("请判断：风险 承压", role="tester"))
        self.assertIn("[规则降级]", payload["reasoning"])

    def test_connection_check_reports_unconfigured_without_network(self):
        cfg = LLMClientConfig(
            api_key="your_api_key_here",
            base_url="https://example.invalid/v1",
            model_name="test",
            stable_mode=True,
            json_retry=1,
            timeout_seconds=5,
            temperature=0.1,
            top_p=0.2,
        )
        status = LLMClient(cfg).check_connection()

        self.assertFalse(status["ok"])
        self.assertEqual(status["status"], "unconfigured")
        self.assertNotIn("api_key", status)

    def test_json_response_format_and_token_cap_are_sent_when_enabled(self):
        cfg = LLMClientConfig(
            api_key="configured",
            base_url="https://example.invalid/v1",
            model_name="test",
            stable_mode=True,
            json_retry=1,
            timeout_seconds=5,
            temperature=0.1,
            top_p=0.2,
            max_tokens=256,
            response_format_json=True,
        )
        client = LLMClient(cfg)
        captured = {}

        def fake_post(payload):
            captured.update(payload)
            return {"choices": [{"message": {"content": '{"answer":"ok","confidence":0.8}'}}]}

        client._post_chat_completion = fake_post
        raw = client.query('请输出 JSON：{"answer":"","confidence":0.0}', role="tester")

        self.assertEqual(json.loads(raw)["answer"], "ok")
        self.assertEqual(captured["max_tokens"], 256)
        self.assertEqual(captured["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
