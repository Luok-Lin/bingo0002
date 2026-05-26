import unittest
from pathlib import Path

import pandas as pd

from dl.kronos_predictor import KronosAdapter


class KronosAdapterTests(unittest.TestCase):
    def test_normalize_ohlcv_accepts_chinese_market_columns(self):
        df = pd.DataFrame(
            [
                {
                    "日期": "2026-05-20",
                    "开盘": "10.1",
                    "最高": "10.5",
                    "最低": "9.9",
                    "收盘": "10.3",
                    "成交量": "100000",
                    "成交额": "1030000",
                },
                {
                    "日期": "2026-05-21",
                    "开盘": "10.3",
                    "最高": "10.8",
                    "最低": "10.2",
                    "收盘": "10.7",
                    "成交量": "120000",
                    "成交额": "1284000",
                },
            ]
        )

        normalized = KronosAdapter.normalize_ohlcv(df)

        self.assertEqual(list(normalized.columns), ["timestamps", "open", "high", "low", "close", "volume", "amount"])
        self.assertEqual(len(normalized), 2)
        self.assertEqual(float(normalized.iloc[-1]["close"]), 10.7)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(normalized["timestamps"]))

    def test_default_source_dir_points_to_bundled_kronos_checkout(self):
        source_dir = Path(KronosAdapter._resolve_source_dir())

        self.assertEqual(source_dir.name, "Kronos")
        self.assertTrue((source_dir / "model" / "__init__.py").exists())


if __name__ == "__main__":
    unittest.main()
