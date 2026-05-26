import json
import os
import tempfile
import unittest

import backend.indexer as indexer


class BackendIndexerTests(unittest.TestCase):
    def test_collect_advice_snapshot_counts_history_and_ignores_invalid_ticker(self):
        original_dir = indexer.ADVICE_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                indexer.ADVICE_DIR = tmpdir
                rows = {
                    "old.json": {
                        "ticker": "600519",
                        "generated_at": "2026-05-19 10:00:00",
                        "recommendation": {"action": "BUY"},
                    },
                    "new.json": {
                        "ticker": "sh600519",
                        "generated_at": "2026-05-20 10:00:00",
                        "recommendation": {"action": "SELL"},
                    },
                    "invalid.json": {
                        "ticker": "",
                        "generated_at": "2026-05-20 11:00:00",
                        "recommendation": {"action": "BUY"},
                    },
                }
                for name, payload in rows.items():
                    with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as f:
                        json.dump(payload, f)

                latest, total = indexer._collect_advice_snapshot()

                self.assertEqual(total, 2)
                self.assertEqual(set(latest.keys()), {"600519"})
                self.assertEqual(latest["600519"]["recommendation"]["action"], "SELL")
            finally:
                indexer.ADVICE_DIR = original_dir

    def test_dashboard_summary_enriches_latest_advice_with_settlement_return(self):
        original_advice_dir = indexer.ADVICE_DIR
        original_settlement_path = indexer.ADVICE_SETTLEMENT_PATH
        original_index_path = indexer.INDEX_SUMMARY_PATH
        original_backtest_dir = indexer.BACKTEST_SUMMARY_DIR
        original_reflections_path = indexer.REFLECTIONS_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                advice_dir = os.path.join(tmpdir, "advice")
                os.makedirs(advice_dir, exist_ok=True)
                settlement_path = os.path.join(tmpdir, "advice_settlements.json")
                index_path = os.path.join(tmpdir, "index_summary.json")
                reflections_path = os.path.join(tmpdir, "reflections.json")
                backtest_dir = os.path.join(tmpdir, "backtests")
                os.makedirs(backtest_dir, exist_ok=True)
                indexer.ADVICE_DIR = advice_dir
                indexer.ADVICE_SETTLEMENT_PATH = settlement_path
                indexer.INDEX_SUMMARY_PATH = index_path
                indexer.REFLECTIONS_PATH = reflections_path
                indexer.BACKTEST_SUMMARY_DIR = backtest_dir

                with open(os.path.join(advice_dir, "603155_latest.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "ticker": "603155",
                            "generated_at": "2026-05-22 14:00:00",
                            "recommendation": {"action": "BUY", "position_percent": 20, "confidence": 0.8},
                        },
                        f,
                    )
                with open(reflections_path, "w", encoding="utf-8") as f:
                    json.dump([], f)
                with open(settlement_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "records": [
                                {
                                    "source": "advice_settlement",
                                    "ticker": "603155",
                                    "decision": "BUY",
                                    "horizon": "T+1",
                                    "horizon_days": 1,
                                    "pnl_percent": 0.8,
                                    "market_move_percent": 4.0,
                                    "excess_return_vs_best_baseline": -0.2,
                                    "as_of_date": "2026-05-22",
                                    "settled_date": "2026-05-25",
                                    "settled_at": "2026-05-25 16:00:00",
                                },
                                {
                                    "source": "advice_settlement",
                                    "ticker": "603155",
                                    "decision": "BUY",
                                    "horizon": "T+5",
                                    "horizon_days": 5,
                                    "pnl_percent": 2.3456,
                                    "market_move_percent": 11.728,
                                    "excess_return_vs_best_baseline": 1.2345,
                                    "as_of_date": "2026-05-22",
                                    "settled_date": "2026-05-29",
                                    "settled_at": "2026-05-29 16:00:00",
                                },
                            ]
                        },
                        f,
                    )

                summary = indexer.build_dashboard_summary()
                row = summary["latest_advice"][0]

                self.assertTrue(row["settled"])
                self.assertEqual(row["settled_horizon"], "T+5")
                self.assertEqual(row["settled_return_pct"], 2.3456)
                self.assertEqual(row["settled_excess_best_pct"], 1.2345)
                self.assertEqual(row["settlement"]["settled_date"], "2026-05-29")
            finally:
                indexer.ADVICE_DIR = original_advice_dir
                indexer.ADVICE_SETTLEMENT_PATH = original_settlement_path
                indexer.INDEX_SUMMARY_PATH = original_index_path
                indexer.BACKTEST_SUMMARY_DIR = original_backtest_dir
                indexer.REFLECTIONS_PATH = original_reflections_path


if __name__ == "__main__":
    unittest.main()
