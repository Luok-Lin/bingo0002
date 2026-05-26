from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _env_int(name: str, default: int, lower: int = 1, upper: int | None = None) -> int:
    try:
        value = max(lower, int(str(os.getenv(name, str(default))).strip() or str(default)))
    except ValueError:
        value = default
    if upper is not None:
        value = min(upper, value)
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
        if num == num:
            return num
    except Exception:
        pass
    return default


class KronosAdapter:
    """Optional Kronos backend with a small interface compatible with DLEngine."""

    REQUIRED_COLUMNS = ("open", "high", "low", "close")

    def __init__(
        self,
        model_name: str | None = None,
        tokenizer_name: str | None = None,
        *,
        pred_len: int | None = None,
        max_context: int | None = None,
        sample_count: int | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name or os.getenv("KRONOS_MODEL", "NeoQuasar/Kronos-small")
        self.tokenizer_name = tokenizer_name or os.getenv("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
        self.pred_len = pred_len or _env_int("KRONOS_PRED_LEN", 5, lower=1, upper=60)
        self.max_context = max_context or _env_int("KRONOS_MAX_CONTEXT", 512, lower=16, upper=2048)
        self.sample_count = sample_count or _env_int("KRONOS_SAMPLE_COUNT", 1, lower=1, upper=20)
        self.device = device or os.getenv("KRONOS_DEVICE", "cpu")
        self.predictor = None
        self.available = False
        self.unavailable_reason = ""
        self.source_dir = self._resolve_source_dir()
        self._load()

    @staticmethod
    def _resolve_source_dir() -> str:
        configured = str(os.getenv("KRONOS_SOURCE_DIR", "") or "").strip()
        if configured:
            return configured
        return str(Path(__file__).resolve().parents[1] / "third_party" / "Kronos")

    def _ensure_source_path(self) -> None:
        source_path = Path(self.source_dir).expanduser().resolve()
        if not source_path.exists():
            return
        source_text = str(source_path)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

    def _load(self) -> None:
        self._ensure_source_path()
        try:
            from model import Kronos, KronosPredictor, KronosTokenizer
        except Exception as exc:
            self.unavailable_reason = f"Kronos package not importable: {exc}"
            return

        try:
            tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
            model = Kronos.from_pretrained(self.model_name)
            self.predictor = KronosPredictor(
                model,
                tokenizer,
                device=self.device,
                max_context=self.max_context,
            )
            self.available = True
            print(f"[Kronos Adapter] 已加载 {self.model_name} / {self.tokenizer_name} ({self.device}).")
        except TypeError:
            try:
                tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
                model = Kronos.from_pretrained(self.model_name)
                self.predictor = KronosPredictor(model, tokenizer, max_context=self.max_context)
                self.available = True
                print(f"[Kronos Adapter] 已加载 {self.model_name} / {self.tokenizer_name}.")
            except Exception as exc:
                self.unavailable_reason = f"Kronos load failed: {exc}"
        except Exception as exc:
            self.unavailable_reason = f"Kronos load failed: {exc}"

    @classmethod
    def normalize_ohlcv(cls, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if "kronos_ohlcv" in data:
                return cls.normalize_ohlcv(data.get("kronos_ohlcv"))
            if "records" in data:
                return cls.normalize_ohlcv(data.get("records"))
            df = pd.DataFrame(data)
        else:
            raise ValueError("Kronos 输入必须是 DataFrame、records 列表或包含 kronos_ohlcv 的 dict。")

        rename_map = {
            "日期": "timestamps",
            "时间": "timestamps",
            "date": "timestamps",
            "datetime": "timestamps",
            "timestamp": "timestamps",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        for col in cls.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Kronos OHLCV 缺少必要字段: {col}")
        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "timestamps" not in df.columns:
            df["timestamps"] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(df), freq="B")
        df["timestamps"] = pd.to_datetime(df["timestamps"], errors="coerce")
        keep_cols = ["timestamps", "open", "high", "low", "close"]
        for optional in ("volume", "amount"):
            if optional in df.columns:
                keep_cols.append(optional)
        df = df[keep_cols].dropna(subset=["timestamps", "open", "high", "low", "close"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("Kronos OHLCV 清洗后为空。")
        return df.tail(_env_int("KRONOS_MAX_CONTEXT", 512, lower=16, upper=2048)).reset_index(drop=True)

    def _future_timestamps(self, timestamps: pd.Series) -> pd.Series:
        last_ts = pd.to_datetime(timestamps.iloc[-1])
        try:
            freq = pd.infer_freq(pd.to_datetime(timestamps.tail(min(len(timestamps), 20))))
        except Exception:
            freq = None
        if freq:
            return pd.Series(pd.date_range(start=last_ts, periods=self.pred_len + 1, freq=freq)[1:])
        return pd.Series([last_ts + timedelta(days=i) for i in range(1, self.pred_len + 1)])

    def predict(self, ticker: str, kronos_ohlcv: Any) -> dict:
        df = self.normalize_ohlcv(kronos_ohlcv)
        if not self.available or self.predictor is None:
            raise RuntimeError(self.unavailable_reason or "Kronos predictor is unavailable.")

        x_df = df[[c for c in ("open", "high", "low", "close", "volume", "amount") if c in df.columns]]
        x_timestamp = df["timestamps"]
        y_timestamp = self._future_timestamps(x_timestamp)

        pred_df = self.predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=self.pred_len,
            T=_safe_float(os.getenv("KRONOS_TEMPERATURE", 1.0), 1.0),
            top_p=_safe_float(os.getenv("KRONOS_TOP_P", 0.9), 0.9),
            sample_count=self.sample_count,
        )
        if pred_df is None or pred_df.empty or "close" not in pred_df.columns:
            raise RuntimeError("Kronos 返回结果为空或缺少 close 字段。")

        last_close = _safe_float(df["close"].iloc[-1])
        forecast_close = _safe_float(pred_df["close"].iloc[-1], last_close)
        forecast_return_pct = ((forecast_close / last_close) - 1.0) * 100.0 if last_close else 0.0
        close_series = pd.to_numeric(pred_df["close"], errors="coerce").dropna()
        step_returns = close_series.pct_change().dropna() * 100.0
        volatility = float(step_returns.std()) if not step_returns.empty else 0.0
        confidence = min(0.95, max(0.05, abs(forecast_return_pct) / max(volatility * 2.0, 1.0)))
        trend = "上涨 (看多)" if forecast_return_pct > 0 else ("下跌 (看空)" if forecast_return_pct < 0 else "震荡 (中性)")
        return {
            "score": round(forecast_return_pct, 4),
            "trend": trend,
            "confidence": f"{round(confidence * 100, 2)}%",
            "model": "kronos",
            "model_name": self.model_name,
            "tokenizer": self.tokenizer_name,
            "horizon": self.pred_len,
            "lookback_rows": int(len(df)),
            "forecast_close": round(forecast_close, 4),
            "forecast_return_pct": round(forecast_return_pct, 4),
            "forecast_volatility_pct": round(volatility, 4),
            "as_of_date": str(df["timestamps"].iloc[-1].date()),
            "target_date": str(pd.to_datetime(y_timestamp.iloc[-1]).date()),
        }
