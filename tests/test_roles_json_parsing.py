import unittest

from agents.roles import enforce_json_contract, parse_llm_json


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


if __name__ == "__main__":
    unittest.main()
