import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import backend.services as services


class AdviceQaTests(unittest.TestCase):
    def _write_advice(self, directory: str, ticker: str = "600519") -> None:
        payload = {
            "ticker": ticker,
            "generated_at": "2026-05-20 10:00:00",
            "as_of_date": "2026-05-20",
            "latest_market": {"close": 100.0, "pct_change": 1.2, "turnover": 2.5},
            "recommendation": {
                "action": "HOLD",
                "execution_action": "HOLD",
                "position_percent": 0.0,
                "confidence": 0.66,
                "reason": "技术面与基本面信号分歧，等待更清晰突破。",
            },
            "analyst_cases": {
                "technical_flow": {
                    "sentiment": "positive",
                    "confidence": 0.7,
                    "reasoning": "放量反弹但尚未突破压力位。",
                },
                "fundamental_news": {
                    "sentiment": "neutral",
                    "confidence": 0.55,
                    "reasoning": "新闻面缺少强催化。",
                },
            },
            "referee": {"decision": "HOLD", "trend_strength": 0.2, "debate_trace": []},
            "risk": {"decision": "HOLD", "position_percent": 0.0, "reason": "风控要求等待确认。"},
            "data_quality": {"score": 0.78, "level": "high", "note": "数据质量较高。"},
            "stability_diagnostics": {"parse_fail_count": 0, "rule_fallback_count": 0, "empty_reason_count": 0},
        }
        with open(os.path.join(directory, f"{ticker}_20260520_100000.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def test_ask_advice_question_uses_latest_advice_context(self):
        with tempfile.TemporaryDirectory() as advice_dir, tempfile.TemporaryDirectory() as state_dir:
            original_advice_dir = services.ADVICE_DIR
            original_history_path = services.ADVICE_QA_HISTORY_PATH
            try:
                services.ADVICE_DIR = advice_dir
                services.ADVICE_QA_HISTORY_PATH = os.path.join(state_dir, "qa_history.json")
                self._write_advice(advice_dir)
                llm_payload = {
                    "answer": "当前建议 HOLD，因为技术面虽有反弹，但基本面缺少催化，风控要求等待确认。",
                    "evidence": [{"source": "最终建议", "text": "技术面与基本面信号分歧"}],
                    "confidence": 0.72,
                    "follow_up_questions": ["什么条件下会转为买入？"],
                    "limitations": "仅基于当前建议文件。",
                }
                with (
                    patch.dict(os.environ, {"ADVICE_QA_USE_LLM": "1"}, clear=False),
                    patch.object(services.LLMClient, "query", return_value=json.dumps(llm_payload, ensure_ascii=False)),
                ):
                    result = services.ask_advice_question(
                        ticker="600519",
                        question="为什么不是买入？",
                        include_rag=False,
                    )

                self.assertEqual(result["ticker"], "600519")
                self.assertIn("HOLD", result["answer"])
                self.assertEqual(result["confidence"], 0.72)
                self.assertGreaterEqual(len(result["evidence"]), 1)
                self.assertTrue(os.path.exists(services.ADVICE_QA_HISTORY_PATH))
            finally:
                services.ADVICE_DIR = original_advice_dir
                services.ADVICE_QA_HISTORY_PATH = original_history_path

    def test_missing_advice_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as advice_dir:
            original_advice_dir = services.ADVICE_DIR
            try:
                services.ADVICE_DIR = advice_dir
                with self.assertRaises(FileNotFoundError):
                    services.ask_advice_question(ticker="600519", question="为什么？", include_rag=False)
            finally:
                services.ADVICE_DIR = original_advice_dir

    def test_rag_request_is_disabled_by_default_env(self):
        with tempfile.TemporaryDirectory() as advice_dir, tempfile.TemporaryDirectory() as state_dir:
            original_advice_dir = services.ADVICE_DIR
            original_history_path = services.ADVICE_QA_HISTORY_PATH
            fake_module = types.ModuleType("scripts.investment_advice")
            fake_module.build_rag_with_diagnostics = Mock()
            try:
                services.ADVICE_DIR = advice_dir
                services.ADVICE_QA_HISTORY_PATH = os.path.join(state_dir, "qa_history.json")
                self._write_advice(advice_dir)
                llm_payload = {
                    "answer": "当前建议仍为 HOLD，追问快答只使用最新建议证据。",
                    "evidence": [{"source": "最终建议", "text": "等待更清晰突破"}],
                    "confidence": 0.61,
                    "follow_up_questions": ["什么时候再检索外部证据？"],
                    "limitations": "未启用追问 RAG。",
                }
                with (
                    patch.dict(os.environ, {"ADVICE_QA_ENABLE_RAG": "0"}, clear=False),
                    patch.dict(os.environ, {"ADVICE_QA_USE_LLM": "1"}, clear=False),
                    patch.dict(sys.modules, {"scripts.investment_advice": fake_module}),
                    patch.object(services.LLMClient, "query", return_value=json.dumps(llm_payload, ensure_ascii=False)),
                ):
                    result = services.ask_advice_question(
                        ticker="600519",
                        question="结合新闻再看一下？",
                        include_rag=True,
                    )

                fake_module.build_rag_with_diagnostics.assert_not_called()
                self.assertEqual(result["rag_status"]["status"], "disabled_by_env")
                self.assertTrue(result["rag_status"]["requested"])
                self.assertFalse(result["rag_status"]["enabled"])
            finally:
                services.ADVICE_DIR = original_advice_dir
                services.ADVICE_QA_HISTORY_PATH = original_history_path

    def test_llm_error_returns_local_fallback(self):
        with tempfile.TemporaryDirectory() as advice_dir, tempfile.TemporaryDirectory() as state_dir:
            original_advice_dir = services.ADVICE_DIR
            original_history_path = services.ADVICE_QA_HISTORY_PATH
            failing_client = Mock()
            failing_client.query.side_effect = TimeoutError("too slow")
            try:
                services.ADVICE_DIR = advice_dir
                services.ADVICE_QA_HISTORY_PATH = os.path.join(state_dir, "qa_history.json")
                self._write_advice(advice_dir)
                with (
                    patch.dict(os.environ, {"ADVICE_QA_USE_LLM": "1"}, clear=False),
                    patch.object(services, "_qa_llm_client", return_value=failing_client),
                ):
                    result = services.ask_advice_question(
                        ticker="600519",
                        question="如果已经持仓怎么办？",
                        include_rag=False,
                    )

                self.assertTrue(result["llm_fallback"])
                self.assertIn("HOLD", result["answer"])
                self.assertIn("本地结构化建议兜底", result["answer"])
            finally:
                services.ADVICE_DIR = original_advice_dir
                services.ADVICE_QA_HISTORY_PATH = original_history_path


if __name__ == "__main__":
    unittest.main()
