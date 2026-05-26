import json
import os
import tempfile
import unittest

import backend.services as services
from backend.reports import build_advice_pdf_bytes


class AdviceReportTests(unittest.TestCase):
    def _write_advice(self, directory: str, ticker: str, stamp: str, generated_at: str, action: str) -> None:
        payload = {
            "ticker": ticker,
            "generated_at": generated_at,
            "as_of_date": "2026-05-23",
            "latest_market": {"close": 10.5, "pct_change": 1.2, "turnover": 2.1},
            "recommendation": {
                "action": action,
                "execution_action": action,
                "position_percent": 20.0,
                "confidence": 0.72,
                "reason": "测试建议理由：量价仍需确认。",
            },
            "risk": {"decision": "HOLD", "position_percent": 20.0, "reason": "跌破关键支撑则减仓。"},
            "data_quality": {"score": 0.8, "level": "high", "note": "测试数据质量良好。"},
            "multi_period_advice": {
                "short_term": {"action": "HOLD", "confidence": 0.6, "reason": "短线观察。"},
                "swing_term": {"action": action, "confidence": 0.7, "reason": "中线等待确认。"},
                "risk_plan": {"stop_loss": "跌破支撑", "take_profit": "冲高分批", "invalid_conditions": ["放量下跌"]},
            },
        }
        with open(os.path.join(directory, f"{ticker}_{stamp}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def test_get_advice_for_report_can_select_specific_generated_at(self):
        with tempfile.TemporaryDirectory() as advice_dir:
            original_advice_dir = services.ADVICE_DIR
            try:
                services.ADVICE_DIR = advice_dir
                self._write_advice(advice_dir, "600519", "20260523_090000", "2026-05-23 09:00:00", "HOLD")
                self._write_advice(advice_dir, "600519", "20260523_100000", "2026-05-23 10:00:00", "BUY")

                latest, latest_id = services.get_advice_for_report("600519")
                specific, specific_id = services.get_advice_for_report("600519", generated_at="2026-05-23 09:00:00")

                self.assertEqual(latest["recommendation"]["action"], "BUY")
                self.assertEqual(latest_id, "600519_20260523_100000")
                self.assertEqual(specific["recommendation"]["action"], "HOLD")
                self.assertEqual(specific_id, "600519_20260523_090000")
            finally:
                services.ADVICE_DIR = original_advice_dir

    def test_build_advice_pdf_returns_pdf_bytes(self):
        advice = {
            "ticker": "600519",
            "generated_at": "2026-05-23 10:00:00",
            "as_of_date": "2026-05-23",
            "recommendation": {"action": "HOLD", "position_percent": 10, "confidence": 0.6, "reason": "测试。"},
            "risk": {"reason": "控制仓位。"},
        }

        pdf = build_advice_pdf_bytes(advice)

        self.assertGreater(len(pdf), 1000)
        self.assertEqual(pdf[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
