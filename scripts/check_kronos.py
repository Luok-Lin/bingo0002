from __future__ import annotations

import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dl.predictor import DLEngine
from scripts.investment_advice import build_technical_feature_bundle


def _sample_bundle() -> dict:
    rows = []
    for idx in range(40):
        close = 10.0 + idx * 0.03
        prev_close = 10.0 + max(idx - 1, 0) * 0.03
        rows.append(
            {
                "日期": f"2026-04-{(idx % 28) + 1:02d}",
                "开盘": close - 0.01,
                "收盘": close,
                "最高": close + 0.05,
                "最低": close - 0.06,
                "成交量": 100_000 + idx * 100,
                "成交额": (100_000 + idx * 100) * close,
                "振幅": 1.0,
                "涨跌幅": (close - prev_close) / prev_close * 100.0 if prev_close else 0.0,
                "涨跌额": close - prev_close,
                "换手率": 1.0,
            }
        )
    return build_technical_feature_bundle(pd.DataFrame(rows))


def main() -> int:
    os.environ.setdefault("DL_BACKEND", "auto")
    os.environ.setdefault("KRONOS_PRED_LEN", "1")
    os.environ.setdefault("KRONOS_SAMPLE_COUNT", "1")
    engine = DLEngine()
    if not engine.kronos or not engine.kronos.available:
        reason = engine.kronos.unavailable_reason if engine.kronos else "Kronos adapter not initialized"
        print(f"Kronos unavailable: {reason}")
        return 1

    pred = engine.predict("600519", _sample_bundle())
    print("Kronos smoke prediction:")
    print(pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
