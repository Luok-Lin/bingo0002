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


if __name__ == "__main__":
    unittest.main()
