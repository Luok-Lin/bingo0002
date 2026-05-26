from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from urllib.parse import quote

import requests


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except Exception:
    pass

urllib.request.getproxies = lambda: {}


def _env_int(name: str, default: int, lower: int = 1) -> int:
    try:
        return max(lower, int(str(os.getenv(name, str(default))).strip() or str(default)))
    except ValueError:
        return default


_AKSHARE_HTTP_TIMEOUT = _env_int("AKSHARE_TIMEOUT_SECONDS", 8, lower=3)
_ORIGINAL_REQUEST = requests.sessions.Session.request


def _request_with_timeout(self, method, url, **kwargs):
    kwargs.setdefault("timeout", _AKSHARE_HTTP_TIMEOUT)
    kwargs.setdefault("proxies", {"http": None, "https": None, "all": None})
    return _ORIGINAL_REQUEST(self, method, url, **kwargs)


if not getattr(requests.sessions.Session.request, "_tradingagents_timeout", False):
    _request_with_timeout._tradingagents_timeout = True
    requests.sessions.Session.request = _request_with_timeout

import akshare as ak
import pandas as pd

from agents.roles import (
    FundamentalNewsAnalyst,
    GameReferee,
    RiskManager,
    TechnicalFlowAnalyst,
    TraderAgent,
)
from memory.memory_bank import MemoryBank
from rag.retriever import SimpleRAG
from backend.strategy_rules import apply_strategy_rules_to_advice


FEATURE_COLUMNS = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
TECHNICAL_FEATURE_VERSION = "technical_features_v1"
KRONOS_OHLCV_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
WEB_INDEX_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "monitoring", "web_index", "dashboard_summary.json")
STABLE_CACHE_DIR = os.path.join(BASE_DIR, "data", "investment_advice", ".stable_cache")
MAX_GENERATION_RETRY = max(1, min(5, int(str(os.getenv("ADVICE_GENERATION_RETRY", "3")).strip() or "3")))
STABLE_CACHE_MINUTES = max(5, min(24 * 60, int(str(os.getenv("STABLE_CACHE_MINUTES", "120")).strip() or "120")))
RULE_FALLBACK_MARKERS = ("[规则降级]", "外部LLM暂不可用")
LLM_ERROR_MARKERS = (
    "CERTIFICATE_VERIFY_FAILED",
    "url error:",
    "http ",
    "json retry exhausted",
    "LLM API key is not configured",
)


def _normalize_action(value: str) -> str:
    token = str(value or "").upper().strip()
    if "BUY" in token:
        return "BUY"
    if "SELL" in token:
        return "SELL"
    return "HOLD"


def _action_from_signal(value: str) -> str:
    token = str(value or "").strip().lower()
    if any(x in token for x in ["positive", "bull", "buy", "看多", "做多", "增持"]):
        return "BUY"
    if any(x in token for x in ["negative", "bear", "sell", "看空", "做空", "减持"]):
        return "SELL"
    return _normalize_action(value)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        num = float(value)
    except Exception:
        num = low
    return max(low, min(high, num))


def _quality_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _days_since_iso(value: str) -> int | None:
    text = str(value or "").strip()[:10]
    if not text:
        return None
    try:
        dt = datetime.strptime(text, "%Y-%m-%d")
        return max(0, (datetime.now() - dt).days)
    except Exception:
        return None


def _date_to_int(value, default: int = 20000101) -> int:
    try:
        return int(str(value)[:10].replace("-", ""))
    except Exception:
        return default


def _emit_progress(
    stage: str,
    percent: int,
    summary: str,
    *,
    status: str = "done",
    title: str = "",
    payload: dict | None = None,
    partial_result: dict | None = None,
) -> None:
    event = {
        "stage": stage,
        "title": title or summary,
        "summary": summary,
        "status": status,
        "progress_percent": max(0, min(100, int(percent))),
    }
    if payload:
        event["payload"] = payload
    if partial_result:
        event["partial_result"] = partial_result
    print(f"TA_PROGRESS {json.dumps(event, ensure_ascii=False)}", flush=True)


def _to_float(value, default: float = 0.0) -> float:
    try:
        num = float(value)
        if num == num:
            return num
    except Exception:
        pass
    return default


def _round_or_none(value, digits: int = 4):
    try:
        num = float(value)
        if num == num:
            return round(num, digits)
    except Exception:
        pass
    return None


def _last_or_none(series: pd.Series, digits: int = 4):
    if series is None or series.empty:
        return None
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return _round_or_none(cleaned.iloc[-1], digits=digits)


def _percentile_rank(series: pd.Series, value) -> float | None:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return None
    try:
        val = float(value)
    except Exception:
        return None
    return round(float((cleaned <= val).mean()), 4)


def _make_signal_text(features: dict) -> list[str]:
    trend = features.get("trend", {}) or {}
    momentum = features.get("momentum", {}) or {}
    risk = features.get("risk", {}) or {}
    volume = features.get("volume", {}) or {}
    signals: list[str] = []

    close = trend.get("close")
    ma5 = trend.get("ma5")
    ma10 = trend.get("ma10")
    ma20 = trend.get("ma20")
    if all(x is not None for x in [close, ma5, ma10, ma20]):
        if close > ma5 > ma10 > ma20:
            signals.append("价格站上 MA5/10/20，短中期均线呈多头排列。")
        elif close < ma5 < ma10 < ma20:
            signals.append("价格跌破 MA5/10/20，短中期均线呈空头排列。")
        elif close > ma20:
            signals.append("价格位于 MA20 上方，但均线结构尚未完全顺排。")
        else:
            signals.append("价格位于 MA20 下方，趋势确认度偏弱。")

    rsi14 = momentum.get("rsi14")
    if rsi14 is not None:
        if rsi14 >= 70:
            signals.append("RSI14 进入偏热区，追高风险上升。")
        elif rsi14 <= 30:
            signals.append("RSI14 进入超卖区，存在修复弹性但需确认。")

    macd_hist = momentum.get("macd_hist")
    if macd_hist is not None:
        signals.append("MACD 柱体为正，动量偏多。" if macd_hist > 0 else "MACD 柱体为负，动量偏空。")

    volume_ratio = volume.get("volume_ratio_5_20")
    if volume_ratio is not None:
        if volume_ratio >= 1.5:
            signals.append("近5日成交量显著高于20日均量，资金活跃度提升。")
        elif volume_ratio <= 0.7:
            signals.append("近5日成交量低于20日均量，资金参与度偏弱。")

    atr_pct = risk.get("atr14_pct")
    if atr_pct is not None and atr_pct >= 5:
        signals.append("ATR14 占比较高，短线波动风险偏大。")
    max_dd = risk.get("max_drawdown60_pct")
    if max_dd is not None and max_dd <= -15:
        signals.append("近60日最大回撤较深，趋势修复需要更多确认。")

    return signals[:8] or ["技术指标未形成明确单边信号。"]


def _format_technical_feature_summary(features: dict) -> str:
    trend = features.get("trend", {}) or {}
    momentum = features.get("momentum", {}) or {}
    risk = features.get("risk", {}) or {}
    volume = features.get("volume", {}) or {}
    signals = features.get("signals", []) or []
    lines = [
        f"技术特征版本: {features.get('version', TECHNICAL_FEATURE_VERSION)}",
        (
            "趋势: "
            f"收盘={trend.get('close')}, MA5={trend.get('ma5')}, MA10={trend.get('ma10')}, "
            f"MA20={trend.get('ma20')}, MA60={trend.get('ma60')}, "
            f"价距MA20={trend.get('price_vs_ma20_pct')}%, MA20五日斜率={trend.get('ma20_slope_5d_pct')}%"
        ),
        (
            "动量: "
            f"5日收益={momentum.get('return_5d_pct')}%, 10日收益={momentum.get('return_10d_pct')}%, "
            f"20日收益={momentum.get('return_20d_pct')}%, RSI14={momentum.get('rsi14')}, "
            f"MACD={momentum.get('macd')}, Signal={momentum.get('macd_signal')}, Hist={momentum.get('macd_hist')}"
        ),
        (
            "波动/风险: "
            f"ATR14占比={risk.get('atr14_pct')}%, 20日年化波动={risk.get('volatility20_annual_pct')}%, "
            f"60日最大回撤={risk.get('max_drawdown60_pct')}%"
        ),
        (
            "成交/换手: "
            f"近5/20日量比={volume.get('volume_ratio_5_20')}, 成交额比={volume.get('amount_ratio_5_20')}, "
            f"最新换手={volume.get('turnover')}%, 60日换手分位={volume.get('turnover_percentile_60')}"
        ),
        "信号摘要: " + " ".join(str(x) for x in signals[:6]),
    ]
    return "\n".join(lines)


def build_technical_feature_bundle(df_hist: pd.DataFrame) -> dict:
    """Build richer technical features for LLM analysts while keeping raw DL matrix compatible."""
    df = df_hist.copy()
    for col in FEATURE_COLUMNS + ["日期"]:
        if col not in df.columns:
            raise ValueError(f"行情数据缺少必要字段: {col}")
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if df.empty:
        raise ValueError("行情数据为空，无法构建技术特征。")

    close = df["收盘"]
    high = df["最高"]
    low = df["最低"]
    volume = df["成交量"]
    amount = df["成交额"]
    turnover = df["换手率"]
    pct = close.pct_change() * 100.0

    ma5 = close.rolling(5, min_periods=1).mean()
    ma10 = close.rolling(10, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()
    ma20_slope_5d = (ma20 / ma20.shift(5) - 1.0) * 100.0

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi14 = 100.0 - (100.0 / (1.0 + rs))
    rsi14 = rsi14.mask((loss == 0) & (gain > 0), 100.0)
    rsi14 = rsi14.mask((loss == 0) & (gain == 0), 50.0)
    rsi14 = rsi14.fillna(50.0)

    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    macd_hist = macd - macd_signal

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()
    atr14_pct = atr14 / close.replace(0, pd.NA) * 100.0
    volatility20_annual = pct.rolling(20, min_periods=2).std() * (252 ** 0.5)
    rolling_high60 = close.rolling(60, min_periods=1).max()
    drawdown60 = close / rolling_high60.replace(0, pd.NA) - 1.0
    max_drawdown60 = drawdown60.rolling(60, min_periods=1).min() * 100.0

    vol_ma5 = volume.rolling(5, min_periods=1).mean()
    vol_ma20 = volume.rolling(20, min_periods=1).mean()
    amount_ma5 = amount.rolling(5, min_periods=1).mean()
    amount_ma20 = amount.rolling(20, min_periods=1).mean()
    latest_turnover = _last_or_none(turnover, digits=4)

    latest_close = _last_or_none(close, digits=4)
    latest_ma20 = _last_or_none(ma20, digits=4)
    feature_values = {
        "version": TECHNICAL_FEATURE_VERSION,
        "as_of_date": str(df.iloc[-1]["日期"]),
        "lookback_rows": int(len(df)),
        "trend": {
            "close": latest_close,
            "ma5": _last_or_none(ma5, digits=4),
            "ma10": _last_or_none(ma10, digits=4),
            "ma20": latest_ma20,
            "ma60": _last_or_none(ma60, digits=4),
            "price_vs_ma20_pct": _round_or_none((latest_close / latest_ma20 - 1.0) * 100.0 if latest_close and latest_ma20 else None),
            "ma20_slope_5d_pct": _last_or_none(ma20_slope_5d, digits=4),
        },
        "momentum": {
            "return_5d_pct": _round_or_none((close.iloc[-1] / close.shift(5).iloc[-1] - 1.0) * 100.0 if len(close) > 5 and close.shift(5).iloc[-1] else None),
            "return_10d_pct": _round_or_none((close.iloc[-1] / close.shift(10).iloc[-1] - 1.0) * 100.0 if len(close) > 10 and close.shift(10).iloc[-1] else None),
            "return_20d_pct": _round_or_none((close.iloc[-1] / close.shift(20).iloc[-1] - 1.0) * 100.0 if len(close) > 20 and close.shift(20).iloc[-1] else None),
            "rsi14": _last_or_none(rsi14, digits=4),
            "macd": _last_or_none(macd, digits=4),
            "macd_signal": _last_or_none(macd_signal, digits=4),
            "macd_hist": _last_or_none(macd_hist, digits=4),
        },
        "risk": {
            "atr14_pct": _last_or_none(atr14_pct, digits=4),
            "volatility20_annual_pct": _last_or_none(volatility20_annual, digits=4),
            "max_drawdown60_pct": _last_or_none(max_drawdown60, digits=4),
        },
        "volume": {
            "volume_ratio_5_20": _round_or_none(vol_ma5.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] else None),
            "amount_ratio_5_20": _round_or_none(amount_ma5.iloc[-1] / amount_ma20.iloc[-1] if amount_ma20.iloc[-1] else None),
            "turnover": latest_turnover,
            "turnover_percentile_60": _percentile_rank(turnover.tail(60), latest_turnover),
        },
    }
    feature_values["signals"] = _make_signal_text(feature_values)
    kronos_ohlcv = pd.DataFrame(
        {
            "timestamps": pd.to_datetime(df["日期"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "open": df["开盘"],
            "high": df["最高"],
            "low": df["最低"],
            "close": df["收盘"],
            "volume": df["成交量"],
            "amount": df["成交额"],
        }
    )
    kronos_ohlcv = kronos_ohlcv.dropna(subset=KRONOS_OHLCV_COLUMNS).tail(512).reset_index(drop=True)
    kronos_records = kronos_ohlcv.to_dict("records")
    feature_values["kronos_ohlcv"] = {
        "columns": KRONOS_OHLCV_COLUMNS,
        "rows": int(len(kronos_records)),
        "max_context": 512,
        "as_of_date": kronos_records[-1]["timestamps"] if kronos_records else "",
    }
    return {
        "dl_features": df.tail(10)[FEATURE_COLUMNS].values,
        "kronos_ohlcv": kronos_records,
        "feature_values": feature_values,
        "summary_text": _format_technical_feature_summary(feature_values),
    }


def fetch_external_knowledge_for_advice(ticker: str, cutoff_date: str | None = None) -> list[dict]:
    """Fetch news/research evidence without importing the backtest entrypoint."""
    import concurrent.futures

    ticker = str(ticker).strip().zfill(6)
    real_news_kb: list[dict] = []
    cutoff_int = _date_to_int(cutoff_date, default=99991231) if cutoff_date else None

    def _exchange_symbol(code: str) -> str:
        return ("SH" if str(code).startswith("6") else "SZ") + str(code).zfill(6)

    def _clean_value(value) -> str:
        text = str(value if value is not None else "").strip()
        if not text or text.lower() in {"nan", "none", "nat"}:
            return ""
        return text

    def _first_existing(row: dict, names: tuple[str, ...]) -> str:
        for name in names:
            if name in row:
                value = _clean_value(row.get(name))
                if value:
                    return value
        return ""

    def _compact_row(row: dict, max_fields: int = 7) -> str:
        preferred = [
            "报告期",
            "公告日期",
            "日期",
            "营业收入",
            "营业总收入",
            "归母净利润",
            "净利润",
            "扣非净利润",
            "经营现金流量净额",
            "资产负债率",
            "净资产收益率",
            "每股收益",
            "毛利率",
            "净利率",
        ]
        pairs = []
        used = set()
        for key in preferred:
            if key in row:
                value = _clean_value(row.get(key))
                if value:
                    pairs.append(f"{key}={value}")
                    used.add(key)
            if len(pairs) >= max_fields:
                break
        if len(pairs) < max_fields:
            for key, value in row.items():
                if key in used:
                    continue
                clean = _clean_value(value)
                if clean:
                    pairs.append(f"{key}={clean}")
                if len(pairs) >= max_fields:
                    break
        return "，".join(pairs)

    def _filter_df_by_cutoff(df, date_columns: tuple[str, ...]):
        if df is None or df.empty or not cutoff_int:
            return df
        try:
            for col in date_columns:
                if col in df.columns:
                    return df[df[col].apply(lambda x: _date_to_int(x) <= cutoff_int)]
        except Exception:
            return df
        return df

    def _append_item(items: list[dict], *, content: str, source: str, date_int: int) -> None:
        content = _clean_value(content)
        if not content:
            return
        if cutoff_int and date_int > cutoff_int:
            return
        items.append(
            {
                "page_content": content,
                "metadata": {"date_int": date_int, "ticker": ticker, "source": source},
            }
        )

    def fetch_news():
        news_items = []
        news_df = ak.stock_news_em(symbol=ticker)
        if news_df is None:
            return news_items, 0
        for _, row in news_df.head(100).iterrows():
            pub_time = str(row.get("发布时间", "2000-01-01"))
            date_int = _date_to_int(pub_time)
            _append_item(
                news_items,
                content="[东方财富新闻] " + str(row.get("新闻标题", "")) + " : " + str(row.get("新闻内容", "")),
                source="news",
                date_int=date_int,
            )
        return news_items, len(news_df)

    def fetch_reports():
        report_items = []
        report_df = ak.stock_research_report_em(symbol=ticker)
        if report_df is not None and not report_df.empty:
            for _, row in report_df.iterrows():
                pub_time = str(row.get("日期", "2000-01-01"))
                date_int = _date_to_int(pub_time)
                content = (
                    f"[券商研报] 机构: {row.get('机构', '未知')} | "
                    f"评级: {row.get('东财评级', '未知')} | 核心观点摘要: {row.get('报告名称', '')}"
                )
                _append_item(report_items, content=content, source="report", date_int=date_int)
        return report_items, 0 if report_df is None else len(report_df)

    def fetch_announcements():
        items = []
        end_date = str(cutoff_date or datetime.now().strftime("%Y-%m-%d"))[:10].replace("-", "")
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=180)).strftime("%Y%m%d")
        notice_df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=ticker,
            market="沪深京",
            start_date=start_date,
            end_date=end_date,
        )
        notice_df = _filter_df_by_cutoff(notice_df, ("公告时间", "公告日期", "日期"))
        if notice_df is not None and not notice_df.empty:
            for _, row in notice_df.head(30).iterrows():
                title = _first_existing(row, ("公告标题", "标题", "公告名称", "简称"))
                pub_time = _first_existing(row, ("公告时间", "公告日期", "日期")) or str(cutoff_date or "")
                date_int = _date_to_int(pub_time)
                _append_item(items, content=f"[公司公告] {title}", source="announcement", date_int=date_int)
        return items, 0 if notice_df is None else len(notice_df)

    def fetch_market_news():
        items = []
        main_df = ak.stock_news_main_cx()
        main_df = _filter_df_by_cutoff(main_df, ("发布时间", "时间", "日期"))
        if main_df is not None and not main_df.empty:
            for _, row in main_df.head(15).iterrows():
                title = _first_existing(row, ("标题", "新闻标题", "内容"))
                pub_time = _first_existing(row, ("发布时间", "时间", "日期")) or str(cutoff_date or "")
                date_int = _date_to_int(pub_time, default=_date_to_int(cutoff_date) if cutoff_date else 20000101)
                _append_item(items, content=f"[市场要闻] {title}", source="market_news", date_int=date_int)
        return items, 0 if main_df is None else len(main_df)

    def fetch_financial_evidence():
        items = []
        source_defs = [
            ("financial_abstract", "财务摘要", lambda: ak.stock_financial_abstract(symbol=ticker), ("报告期", "公告日期", "日期")),
            ("financial_indicator", "财务指标", lambda: ak.stock_financial_analysis_indicator(symbol=ticker, start_year=str(datetime.now().year - 5)), ("日期", "报告期", "公告日期")),
            ("income_statement", "利润表", lambda: ak.stock_profit_sheet_by_report_em(symbol=_exchange_symbol(ticker)), ("公告日期", "REPORT_DATE", "报告期", "日期")),
            ("balance_sheet", "资产负债表", lambda: ak.stock_balance_sheet_by_report_em(symbol=_exchange_symbol(ticker)), ("公告日期", "REPORT_DATE", "报告期", "日期")),
            ("cash_flow", "现金流量表", lambda: ak.stock_cash_flow_sheet_by_report_em(symbol=_exchange_symbol(ticker)), ("公告日期", "REPORT_DATE", "报告期", "日期")),
        ]
        total_rows = 0
        for source, label, loader, date_cols in source_defs:
            try:
                df = loader()
                df = _filter_df_by_cutoff(df, date_cols)
                if df is None or df.empty:
                    continue
                total_rows += len(df)
                row = df.tail(1).to_dict("records")[0]
                pub_time = _first_existing(row, date_cols) or str(cutoff_date or "")
                date_int = _date_to_int(pub_time, default=_date_to_int(cutoff_date) if cutoff_date else 20000101)
                _append_item(items, content=f"[{label}] {_compact_row(row)}", source=source, date_int=date_int)
            except Exception as exc:
                print(f"{label}提取受限: {exc}")
        return items, total_rows

    fetchers = [
        ("新闻", fetch_news),
        ("研报", fetch_reports),
        ("公告", fetch_announcements),
        ("市场要闻", fetch_market_news),
        ("财务证据", fetch_financial_evidence),
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fetchers)) as executor:
        futures = {executor.submit(fetcher): label for label, fetcher in fetchers}
        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                items, count = future.result()
                real_news_kb.extend(items)
                print(f"成功攫取 {count} 条/份 {label} 证据。")
            except Exception as exc:
                print(f"{label}抓取受限: {exc}")

    return real_news_kb


def _stable_cache_path(ticker: str, as_of_date: str) -> str:
    safe_date = str(as_of_date or "").replace("-", "")
    return os.path.join(STABLE_CACHE_DIR, f"{str(ticker).zfill(6)}_{safe_date}.json")


def _load_latest_advice_payload(ticker: str) -> dict:
    ticker = str(ticker).strip().zfill(6)
    advice_dir = os.path.join(BASE_DIR, "data", "investment_advice")
    if not os.path.isdir(advice_dir):
        return {}
    candidates = []
    for name in os.listdir(advice_dir):
        if not (name.startswith(f"{ticker}_") and name.endswith(".json")):
            continue
        full = os.path.join(advice_dir, name)
        if os.path.isfile(full):
            candidates.append((os.path.getmtime(full), full))
    candidates.sort(reverse=True)
    for _, path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and str(payload.get("ticker", "")).zfill(6) == ticker:
                return payload
        except Exception:
            continue
    return {}


def _load_stable_cache(ticker: str, as_of_date: str) -> dict:
    path = _stable_cache_path(ticker, as_of_date)
    if not os.path.exists(path):
        return {}
    age_seconds = time.time() - os.path.getmtime(path)
    if age_seconds > STABLE_CACHE_MINUTES * 60:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if str(payload.get("ticker", "")).zfill(6) != str(ticker).zfill(6):
            return {}
        if str(payload.get("as_of_date", "")) != str(as_of_date):
            return {}
        return payload
    except Exception:
        return {}


def _write_stable_cache(advice: dict) -> None:
    if not _is_stable_cacheable_advice(advice):
        return
    try:
        os.makedirs(STABLE_CACHE_DIR, exist_ok=True)
        path = _stable_cache_path(advice.get("ticker", ""), advice.get("as_of_date", ""))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(advice, f, ensure_ascii=False, indent=2)
    except Exception:
        # 缓存失败不影响主流程
        pass


def _collect_reasoning_texts(advice: dict) -> list[str]:
    out = []
    analyst_cases = advice.get("analyst_cases", {}) or {}
    for _, case in analyst_cases.items():
        if isinstance(case, dict):
            out.append(str(case.get("reasoning", "") or ""))
            out.append(str(case.get("thought_process", "") or ""))
            for src in case.get("source_reports", []) or []:
                if isinstance(src, dict):
                    out.append(str(src.get("reasoning", "") or ""))
                    out.append(str(src.get("thought_process", "") or ""))
    referee = advice.get("referee", {}) or {}
    out.append(str(referee.get("reason", "") or ""))
    for item in referee.get("debate_trace", []) or []:
        judge = (item or {}).get("judge", {}) or {}
        out.append(str(judge.get("reasoning", "") or ""))
    risk = advice.get("risk", {}) or {}
    out.append(str(risk.get("reason", "") or ""))
    rec = advice.get("recommendation", {}) or {}
    out.append(str(rec.get("reason", "") or ""))
    return [x for x in out if str(x).strip()]


def _contains_rule_fallback_text(text: str) -> bool:
    raw = str(text or "")
    return any(marker in raw for marker in RULE_FALLBACK_MARKERS)


def _contains_llm_error_text(text: str) -> bool:
    raw = str(text or "")
    return any(marker in raw for marker in LLM_ERROR_MARKERS)


def _has_rule_fallback(advice: dict) -> bool:
    diagnostics = advice.get("stability_diagnostics", {}) or {}
    try:
        if int(diagnostics.get("rule_fallback_count", 0) or 0) > 0:
            return True
    except Exception:
        pass

    texts = _collect_reasoning_texts(advice)
    for text in texts:
        if _contains_rule_fallback_text(text) or _contains_llm_error_text(text):
            return True
    return False


def _is_stable_cacheable_advice(advice: dict) -> bool:
    return not _has_rule_fallback(advice)


def _cache_matches_current_dl_backend(advice: dict) -> bool:
    backend = str(os.getenv("DL_BACKEND", "lstm") or "lstm").strip().lower()
    if backend not in {"kronos", "auto"}:
        return True
    technical = ((advice.get("analyst_cases", {}) or {}).get("technical_flow", {}) or {})
    for report in technical.get("source_reports", []) or []:
        if not isinstance(report, dict):
            continue
        if str(report.get("model", "")).lower() == "kronos":
            return True
        prediction = report.get("prediction", {}) if isinstance(report.get("prediction"), dict) else {}
        if str(prediction.get("model", "")).lower() == "kronos":
            return True
    return False


def _collect_parse_failures(advice: dict) -> list[dict]:
    failures: list[dict] = []

    def visit(obj, path: str) -> None:
        if isinstance(obj, dict):
            if obj.get("_parse_ok") is False:
                failures.append(
                    {
                        "path": path or "root",
                        "agent": obj.get("agent", ""),
                        "error": str(obj.get("_parse_error", "") or "")[:160],
                    }
                )
            for key, value in obj.items():
                visit(value, f"{path}.{key}" if path else str(key))
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                visit(value, f"{path}[{idx}]")

    visit(advice.get("analyst_cases", {}) or {}, "analyst_cases")
    visit(advice.get("referee", {}) or {}, "referee")
    visit(advice.get("risk", {}) or {}, "risk")
    return failures


def _compute_stability(advice: dict) -> dict:
    technical = ((advice.get("analyst_cases", {}) or {}).get("technical_flow", {}) or {})
    fundamental = ((advice.get("analyst_cases", {}) or {}).get("fundamental_news", {}) or {})
    referee = advice.get("referee", {}) or {}
    recommendation = advice.get("recommendation", {}) or {}
    risk = advice.get("risk", {}) or {}

    tech_sent = str(technical.get("sentiment", "neutral")).lower()
    fund_sent = str(fundamental.get("sentiment", "neutral")).lower()
    ref_decision = _normalize_action(referee.get("decision"))
    rec_action = _normalize_action(recommendation.get("action"))
    risk_action = _normalize_action(risk.get("decision") or risk.get("action") or ref_decision)

    tech_conf = float(technical.get("confidence", 0.5) or 0.5)
    fund_conf = float(fundamental.get("confidence", 0.5) or 0.5)
    ref_conf = float(referee.get("confidence", recommendation.get("confidence", 0.5)) or 0.5)

    score = 0.45
    score += 0.2 * max(0.0, min(1.0, (tech_conf + fund_conf) / 2))
    score += 0.25 * max(0.0, min(1.0, ref_conf))
    if tech_sent == fund_sent:
        score += 0.1
    if rec_action == ref_decision:
        score += 0.08
    if rec_action == risk_action:
        score += 0.05

    texts = _collect_reasoning_texts(advice)
    parse_failures = _collect_parse_failures(advice)
    parse_fail_count = len(parse_failures)
    fallback_count = sum(1 for t in texts if "[规则降级]" in t)
    empty_reason_count = 0
    for path in [
        ("recommendation", "reason"),
        ("risk", "reason"),
        ("analyst_cases", "technical_flow", "reasoning"),
        ("analyst_cases", "fundamental_news", "reasoning"),
    ]:
        obj = advice
        for key in path[:-1]:
            obj = obj.get(key, {}) if isinstance(obj, dict) else {}
        val = obj.get(path[-1], "") if isinstance(obj, dict) else ""
        if not str(val or "").strip():
            empty_reason_count += 1

    score -= min(0.25, 0.05 * parse_fail_count)
    score -= min(0.25, 0.06 * fallback_count)
    score -= min(0.2, 0.06 * empty_reason_count)
    score = max(0.0, min(1.0, round(score, 4)))

    if score >= 0.75:
        level = "high"
        note = "稳定度较高。"
    elif score >= 0.55:
        level = "medium"
        note = "稳定度中等，建议结合盘中验证。"
    else:
        level = "low"
        note = "仅供参考，建议复核。"

    return {
        "stability_score": score,
        "stability_level": level,
        "stability_note": note,
        "stability_diagnostics": {
            "parse_fail_count": parse_fail_count,
            "parse_failures": parse_failures[:8],
            "rule_fallback_count": fallback_count,
            "empty_reason_count": empty_reason_count,
            "consensus_tech_fund": tech_sent == fund_sent,
            "referee_action": ref_decision,
            "risk_action": risk_action,
            "recommendation_action": rec_action,
        },
    }


def _assess_analyst_evidence(technical_case: dict, fundamental_case: dict) -> dict:
    reports: list[dict] = []
    for case in [technical_case or {}, fundamental_case or {}]:
        for report in case.get("source_reports", []) or []:
            if isinstance(report, dict):
                reports.append(report)

    parse_fail_count = sum(1 for r in reports if r.get("_parse_ok") is False)
    low_confidence_count = 0
    neutral_count = 0
    for item in [technical_case or {}, fundamental_case or {}, *reports]:
        try:
            if float(item.get("confidence", 0.5) or 0.5) < 0.3:
                low_confidence_count += 1
        except Exception:
            pass
        if str(item.get("sentiment", "")).lower() == "neutral":
            neutral_count += 1

    return {
        "source_report_count": len(reports),
        "parse_fail_count": parse_fail_count,
        "low_confidence_count": low_confidence_count,
        "neutral_count": neutral_count,
    }


def assess_data_quality(
    *,
    market_diagnostics: dict,
    rag_diagnostics: dict,
    technical_case: dict | None = None,
    fundamental_case: dict | None = None,
) -> dict:
    market = dict(market_diagnostics or {})
    rag = dict(rag_diagnostics or {})
    analyst = _assess_analyst_evidence(technical_case or {}, fundamental_case or {})
    diagnostics: list[str] = []

    rows = int(market.get("rows") or 0)
    market_score = 0.25 + min(0.45, rows / 260.0 * 0.45)
    if rows >= 120:
        market_score += 0.15
    elif rows < 60:
        diagnostics.append("行情样本少于60条")
    freshness_days = market.get("freshness_days")
    if freshness_days is not None:
        if freshness_days <= 5:
            market_score += 0.1
        elif freshness_days > 30:
            market_score -= 0.12
            diagnostics.append("行情日期偏旧")
    if market.get("fallback_used"):
        market_score -= 0.25
        if str(market.get("source", "")) == "marketless_direct_fallback":
            diagnostics.append("行情数据不可用，使用无行情直接兜底")
        else:
            diagnostics.append("行情使用本地回测兜底")
    market_score = _clamp(market_score)

    doc_count = int(rag.get("documents") or 0)
    source_types = rag.get("source_types") or {}
    real_sources = {k: v for k, v in source_types.items() if k != "fallback" and int(v or 0) > 0}
    rag_score = min(0.6, doc_count / 8.0 * 0.6)
    rag_score += min(0.2, len(real_sources) * 0.1)
    rag_freshness_days = rag.get("freshness_days")
    if rag_freshness_days is not None and not rag.get("fallback_used"):
        if rag_freshness_days <= 30:
            rag_score += 0.1
        elif rag_freshness_days > 90:
            rag_score -= 0.1
            diagnostics.append("新闻研报证据偏旧")
    if rag.get("fallback_used"):
        rag_score = min(rag_score, 0.25)
        diagnostics.append("新闻研报使用兜底文本")
    if rag.get("vectorstore_degraded"):
        rag_score = max(0.0, rag_score - 0.08)
        diagnostics.append("向量检索降级为内存关键词检索")
    rag_score = _clamp(rag_score)

    analyst_score = 0.75
    analyst_score -= min(0.24, analyst["parse_fail_count"] * 0.08)
    analyst_score -= min(0.24, analyst["low_confidence_count"] * 0.06)
    analyst_score -= min(0.18, analyst["neutral_count"] * 0.03)
    if analyst["parse_fail_count"]:
        diagnostics.append("存在LLM结构化解析失败")
    if analyst["low_confidence_count"]:
        diagnostics.append("存在低置信度子模块")
    analyst_score = _clamp(analyst_score)

    score = round(_clamp(market_score * 0.45 + rag_score * 0.35 + analyst_score * 0.20), 4)
    if not diagnostics:
        diagnostics.append("主要数据链路正常")
    note_map = {
        "high": "数据质量较高，证据覆盖可支撑当前建议。",
        "medium": "数据质量中等，建议结合盘口与后续公告复核。",
        "low": "数据质量偏低，已降低建议置信度与仓位权重。",
    }
    level = _quality_level(score)
    return {
        "score": score,
        "level": level,
        "note": note_map[level],
        "market": market,
        "rag": rag,
        "analyst": analyst,
        "diagnostics": diagnostics[:6],
        "components": {
            "market_score": round(market_score, 4),
            "rag_score": round(rag_score, 4),
            "analyst_score": round(analyst_score, 4),
        },
    }


def calibrate_recommendation_by_quality(recommendation: dict, risk: dict, data_quality: dict) -> dict:
    calibrated = dict(recommendation or {})
    score = _clamp((data_quality or {}).get("score", 0.5))
    action = _normalize_action(calibrated.get("action"))
    original_confidence = float(calibrated.get("confidence", 0.5) or 0.5)
    original_position = float(calibrated.get("position_percent", 0.0) or 0.0)

    confidence_cap = 1.0
    position_cap = 100.0
    adjustments: list[str] = []
    if score < 0.45:
        confidence_cap = 0.55
        position_cap = 20.0
        adjustments.append("数据质量低，置信度上限0.55、方向性仓位上限20%。")
    elif score < 0.60:
        confidence_cap = 0.68
        position_cap = 35.0
        adjustments.append("数据质量中低，置信度上限0.68、方向性仓位上限35%。")

    if adjustments and action in {"BUY", "SELL"}:
        calibrated["confidence"] = round(min(original_confidence, confidence_cap), 4)
        calibrated["position_percent"] = round(min(original_position, position_cap), 2)
        if calibrated["position_percent"] <= 0:
            action = "HOLD"
            calibrated["action"] = action
            adjustments.append("方向性仓位为0，执行动作转为HOLD。")
        calibrated["execution_action"] = (
            f"{action} {calibrated['position_percent']}%"
            if action in {"BUY", "SELL"} and calibrated["position_percent"] > 0
            else action
        )
        if risk is not None:
            risk["position_percent"] = calibrated["position_percent"]
            risk["decision"] = action
            risk["action"] = action
            risk["final_action"] = action
            risk["quality_adjusted"] = True
            risk["reason"] = f"{risk.get('reason', '')} | 数据质量校准: {' '.join(adjustments)}".strip()
    else:
        calibrated["confidence"] = round(original_confidence, 4)
        calibrated["position_percent"] = round(original_position, 2)

    note = (data_quality or {}).get("note", "")
    if adjustments:
        note = f"{note} {' '.join(adjustments)}".strip()
    calibrated["data_quality_note"] = note
    calibrated["quality_adjustments"] = adjustments
    return calibrated


def _direction_score(action: str) -> float:
    normalized = _normalize_action(action)
    if normalized == "BUY":
        return 1.0
    if normalized == "SELL":
        return -1.0
    return 0.0


def _score_to_action(score: float, buy_threshold: float = 0.18, sell_threshold: float = -0.18) -> str:
    if score >= buy_threshold:
        return "BUY"
    if score <= sell_threshold:
        return "SELL"
    return "HOLD"


def _confidence_from_score(score: float, base_confidence: float, quality_score: float) -> float:
    confidence = 0.42 + abs(score) * 0.32 + _clamp(base_confidence) * 0.22 + _clamp(quality_score) * 0.12
    if quality_score < 0.45:
        confidence = min(confidence, 0.55)
    elif quality_score < 0.60:
        confidence = min(confidence, 0.68)
    return round(_clamp(confidence, 0.35, 0.92), 2)


def _first_matching_signal(signals: list[str], keywords: tuple[str, ...], fallback: str) -> str:
    for signal in signals:
        text = str(signal or "")
        if any(keyword in text for keyword in keywords):
            return text
    return fallback


def _period_reason_for_action(
    *,
    action: str,
    score: float,
    return_5d: float,
    return_20d: float,
    volume_ratio: float,
    price_vs_ma20: float,
    ma20_slope: float,
    signals: list[str],
    horizon: str,
) -> str:
    normalized = _normalize_action(action)
    if horizon == "short":
        if normalized == "BUY":
            return _first_matching_signal(
                signals,
                ("动量偏多", "资金活跃度提升", "多头排列", "MA20 上方"),
                f"短线分数 {score:.2f}，5日收益 {return_5d:.2f}%，量比 {volume_ratio:.2f}，短线动量偏多。",
            )
        if normalized == "SELL":
            return _first_matching_signal(
                signals,
                ("跌破", "偏热", "波动风险", "回撤较深", "空头排列"),
                f"短线分数 {score:.2f}，5日收益 {return_5d:.2f}%，量比 {volume_ratio:.2f}，短线回撤与波动风险占优。",
            )
        return f"短线分数 {score:.2f}，5日收益 {return_5d:.2f}%，量比 {volume_ratio:.2f}，多空信号互相抵消，先观察确认。"

    if normalized == "BUY":
        return _first_matching_signal(
            signals,
            ("MA20 上方", "多头排列", "趋势", "动量偏多"),
            f"中线分数 {score:.2f}，价距MA20 {price_vs_ma20:.2f}%，MA20斜率 {ma20_slope:.2f}%，中线结构偏多。",
        )
    if normalized == "SELL":
        return _first_matching_signal(
            signals,
            ("跌破", "最大回撤", "空头排列", "趋势确认度偏弱"),
            f"中线分数 {score:.2f}，20日收益 {return_20d:.2f}%，价距MA20 {price_vs_ma20:.2f}%，中线风险占优。",
        )
    return f"中线分数 {score:.2f}，价距MA20 {price_vs_ma20:.2f}%，MA20斜率 {ma20_slope:.2f}%，趋势未给出新的加减仓确认。"


def build_multi_period_advice(
    recommendation: dict,
    risk: dict,
    referee: dict,
    analyst_cases: dict,
    technical_features: dict,
    data_quality: dict,
) -> dict:
    """Build practical short-term, swing-term and risk-plan views from the same evidence stack."""
    rec = recommendation or {}
    risk = risk or {}
    referee = referee or {}
    analysts = analyst_cases or {}
    features = technical_features or {}
    trend = features.get("trend", {}) if isinstance(features, dict) else {}
    momentum = features.get("momentum", {}) if isinstance(features, dict) else {}
    risk_features = features.get("risk", {}) if isinstance(features, dict) else {}
    volume = features.get("volume", {}) if isinstance(features, dict) else {}
    signals = [str(x) for x in (features.get("signals", []) if isinstance(features, dict) else []) if str(x or "").strip()]

    rec_action = _normalize_action(rec.get("action"))
    ref_action = _normalize_action(referee.get("decision") or rec_action)
    tech_action = _action_from_signal((analysts.get("technical_flow", {}) or {}).get("sentiment"))
    fund_action = _action_from_signal((analysts.get("fundamental_news", {}) or {}).get("sentiment"))
    rec_confidence = _clamp(rec.get("confidence", 0.5))
    quality_score = _clamp((data_quality or {}).get("score", 0.5))
    trend_strength = _clamp(referee.get("trend_strength", 0.0))

    return_5d = _to_float(momentum.get("return_5d_pct"), 0.0)
    return_20d = _to_float(momentum.get("return_20d_pct"), 0.0)
    rsi14 = _to_float(momentum.get("rsi14"), 50.0)
    macd_hist = _to_float(momentum.get("macd_hist"), 0.0)
    volume_ratio = _to_float(volume.get("volume_ratio_5_20"), 1.0)
    price_vs_ma20 = _to_float(trend.get("price_vs_ma20_pct"), 0.0)
    ma20_slope = _to_float(trend.get("ma20_slope_5d_pct"), 0.0)

    short_score = 0.0
    short_score += _direction_score(tech_action) * 0.20
    short_score += _direction_score(ref_action) * 0.16
    short_score += _clamp(return_5d / 8.0, -0.28, 0.28)
    short_score += 0.12 if macd_hist > 0 else (-0.12 if macd_hist < 0 else 0.0)
    if volume_ratio >= 1.3:
        short_score += 0.10 if return_5d >= 0 else -0.08
    elif volume_ratio <= 0.75:
        short_score -= 0.06
    if rsi14 >= 74:
        short_score -= 0.16
    elif 52 <= rsi14 <= 68:
        short_score += 0.08
    elif rsi14 <= 30:
        short_score += 0.05
    short_score = _clamp(short_score, -1.0, 1.0)

    swing_score = 0.0
    swing_score += _direction_score(rec_action) * 0.26
    swing_score += _direction_score(ref_action) * (0.20 + trend_strength * 0.12)
    swing_score += _direction_score(fund_action) * 0.18
    swing_score += _clamp(price_vs_ma20 / 8.0, -0.18, 0.18)
    swing_score += _clamp(ma20_slope / 5.0, -0.12, 0.12)
    swing_score += _clamp(return_20d / 15.0, -0.16, 0.16)
    if rsi14 >= 78:
        swing_score -= 0.08
    if quality_score < 0.45:
        swing_score *= 0.72
    swing_score = _clamp(swing_score, -1.0, 1.0)

    short_action = _score_to_action(short_score, buy_threshold=0.22, sell_threshold=-0.20)
    swing_action = _score_to_action(swing_score, buy_threshold=0.18, sell_threshold=-0.18)
    short_confidence = _confidence_from_score(short_score, rec_confidence, quality_score)
    swing_confidence = _confidence_from_score(swing_score, rec_confidence, quality_score)

    short_reason = _period_reason_for_action(
        action=short_action,
        score=short_score,
        return_5d=return_5d,
        return_20d=return_20d,
        volume_ratio=volume_ratio,
        price_vs_ma20=price_vs_ma20,
        ma20_slope=ma20_slope,
        signals=signals,
        horizon="short",
    )
    swing_reason = _period_reason_for_action(
        action=swing_action,
        score=swing_score,
        return_5d=return_5d,
        return_20d=return_20d,
        volume_ratio=volume_ratio,
        price_vs_ma20=price_vs_ma20,
        ma20_slope=ma20_slope,
        signals=signals,
        horizon="swing",
    )
    if short_action == "HOLD" and rec_action == "BUY":
        short_reason = f"{short_reason} 短线未给出追买确认，先等回踩或放量突破。"
    if swing_action == "BUY" and short_action != "BUY":
        swing_reason = f"{swing_reason} 中线仍偏多，但买点节奏需等短线确认。"
    if swing_action == "SELL" and rec_action != "SELL":
        swing_reason = f"{swing_reason} 中线结构转弱，持仓应降低预期。"

    close = _to_float(trend.get("close"), 0.0)
    ma20 = _to_float(trend.get("ma20"), 0.0)
    atr_pct = max(1.5, _to_float(risk_features.get("atr14_pct"), 3.0))
    stop_pct = round(max(3.0, min(10.0, atr_pct * 1.25)), 2)
    take_pct = round(max(6.0, min(18.0, atr_pct * 2.2)), 2)
    stop_price = round(close * (1.0 - stop_pct / 100.0), 2) if close else None
    take_price = round(close * (1.0 + take_pct / 100.0), 2) if close else None
    ma20_text = f"或收盘跌破20日线 {ma20:.2f}" if ma20 else "或收盘跌破20日线"
    if swing_action == "BUY" or rec_action == "BUY":
        stop_loss = f"跌破 {stop_price:.2f}（约-{stop_pct:.2f}%）{ma20_text} 时减仓/止损。" if stop_price else f"跌破现价约 {stop_pct:.2f}% {ma20_text} 时减仓/止损。"
        take_profit = f"上涨至 {take_price:.2f}（约+{take_pct:.2f}%）或RSI持续过热时分批止盈。" if take_price else f"上涨约 {take_pct:.2f}% 或RSI持续过热时分批止盈。"
    elif swing_action == "SELL" or rec_action == "SELL":
        stop_loss = "已有持仓优先减仓；若重新站回20日线且量能修复，停止继续看空。"
        take_profit = f"若回撤扩大至约 {stop_pct:.2f}% 后出现放量修复，空头/减仓收益先兑现。"
    else:
        stop_loss = f"未入场不追高；已有持仓跌破 {stop_price:.2f} 或20日线时降仓。" if stop_price else "未入场不追高；已有持仓跌破20日线时降仓。"
        take_profit = f"若放量突破并达到约 +{take_pct:.2f}% 的短线弹性，先锁定部分收益。"

    invalid_conditions = [
        "收盘跌破20日线" if ma20 else "关键均线失守",
        "成交量连续萎缩且无法放量修复",
        "技术分析师与基本面分析师同时转为空头",
    ]
    if quality_score < 0.60:
        invalid_conditions.append("数据质量继续下降或关键行情/研报缺失")
    if rec_action == "BUY":
        invalid_conditions.append("买入后未能延续强于大盘的相对表现")

    return {
        "short_term": {
            "horizon": "1-3 trading days",
            "action": short_action,
            "confidence": short_confidence,
            "reason": short_reason,
        },
        "swing_term": {
            "horizon": "2-4 weeks",
            "action": swing_action,
            "confidence": swing_confidence,
            "reason": swing_reason,
        },
        "risk_plan": {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "invalid_conditions": invalid_conditions[:5],
        },
    }


def _validate_advice_payload(advice: dict) -> list[str]:
    errors: list[str] = []
    rec = advice.get("recommendation", {}) or {}
    risk = advice.get("risk", {}) or {}
    referee = advice.get("referee", {}) or {}
    analysts = advice.get("analyst_cases", {}) or {}
    technical = analysts.get("technical_flow", {}) or {}
    fundamental = analysts.get("fundamental_news", {}) or {}
    multi_period = advice.get("multi_period_advice", {}) or {}
    short_term = multi_period.get("short_term") or advice.get("short_term") or {}
    swing_term = multi_period.get("swing_term") or advice.get("swing_term") or {}
    risk_plan = multi_period.get("risk_plan") or advice.get("risk_plan") or {}

    if _normalize_action(rec.get("action")) not in {"BUY", "SELL", "HOLD"}:
        errors.append("recommendation.action 缺失或非法")
    if not str(rec.get("reason", "") or "").strip():
        errors.append("recommendation.reason 为空")
    if not str(risk.get("reason", "") or "").strip():
        errors.append("risk.reason 为空")
    if _normalize_action(referee.get("decision")) not in {"BUY", "SELL", "HOLD"}:
        errors.append("referee.decision 缺失或非法")
    if not str(technical.get("reasoning", "") or "").strip():
        errors.append("technical_flow.reasoning 为空")
    if not str(fundamental.get("reasoning", "") or "").strip():
        errors.append("fundamental_news.reasoning 为空")
    for key, item in (("short_term", short_term), ("swing_term", swing_term)):
        if not isinstance(item, dict):
            errors.append(f"{key} 缺失或非法")
            continue
        if _normalize_action(item.get("action")) not in {"BUY", "SELL", "HOLD"}:
            errors.append(f"{key}.action 缺失或非法")
        if not str(item.get("reason", "") or "").strip():
            errors.append(f"{key}.reason 为空")
    if not isinstance(risk_plan, dict):
        errors.append("risk_plan 缺失或非法")
    else:
        if not str(risk_plan.get("stop_loss", "") or "").strip():
            errors.append("risk_plan.stop_loss 为空")
        if not str(risk_plan.get("take_profit", "") or "").strip():
            errors.append("risk_plan.take_profit 为空")
        if not isinstance(risk_plan.get("invalid_conditions"), list) or not risk_plan.get("invalid_conditions"):
            errors.append("risk_plan.invalid_conditions 为空")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured investment advice for a given stock.")
    parser.add_argument("ticker", help="A股股票代码，例如 600519")
    parser.add_argument("--debate-depth", type=int, default=2, help="裁判与两个综合分析师最大博弈轮数")
    parser.add_argument("--output-path", default=None, help="结构化建议 JSON 输出路径")
    parser.add_argument("--human-comment", default="", help="人工评论，将作为经验沉淀写入记忆库")
    parser.add_argument("--human-decision", default=None, help="人工修正最终方向，可选 BUY/SELL/HOLD")
    parser.add_argument("--skip-strategy-rules", action="store_true", help="跳过系统策略规则库校准")
    return parser.parse_args()


def fetch_history_with_diagnostics(ticker: str) -> tuple[pd.DataFrame, dict]:
    ticker = str(ticker).strip().zfill(6)

    def _finalize(df_raw: pd.DataFrame, *, turnover_multiplier: float) -> pd.DataFrame:
        df_hist = df_raw.copy()
        df_hist["日期"] = pd.to_datetime(df_hist["日期"]).dt.strftime("%Y-%m-%d")
        df_hist["前收盘"] = df_hist["收盘"].shift(1)
        df_hist["涨跌额"] = df_hist["收盘"] - df_hist["前收盘"]
        df_hist["涨跌幅"] = df_hist["涨跌额"] / df_hist["前收盘"] * 100
        df_hist["振幅"] = (df_hist["最高"] - df_hist["最低"]) / df_hist["前收盘"] * 100
        df_hist["换手率"] = df_hist["换手率"] * turnover_multiplier
        df_hist.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
        df_hist.dropna(subset=["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"], inplace=True)
        df_hist.reset_index(drop=True, inplace=True)
        return df_hist.tail(260).reset_index(drop=True)

    def _fallback_from_local_backtests() -> pd.DataFrame:
        if not os.path.exists(WEB_INDEX_SUMMARY_PATH):
            raise RuntimeError("本地兜底数据缺失: dashboard_summary.json 不存在")
        with open(WEB_INDEX_SUMMARY_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        runs = payload.get("recent_backtest_runs", []) or []
        target = None
        for run in runs:
            if str(run.get("ticker", "")).zfill(6) == ticker:
                target = run
                break
        if not target:
            raise RuntimeError(f"本地兜底数据缺失: 未找到 {ticker} 的回测记录")
        rows = target.get("rows", []) or []
        if len(rows) < 20:
            raise RuntimeError(f"本地兜底数据不足: {ticker} 回测样本少于 20 条")

        synth = []
        prev_close = 100.0
        for idx, row in enumerate(rows):
            date = row.get("date") or row.get("next_date") or f"day_{idx+1}"
            real_pnl = float(row.get("real_pnl", 0.0) or 0.0) / 100.0
            eff_pnl = float(row.get("effective_pnl", 0.0) or 0.0) / 100.0
            open_px = prev_close
            close_px = max(0.1, open_px * (1 + real_pnl))
            wick = max(abs(close_px - open_px) * 0.6, open_px * (0.004 + min(abs(eff_pnl), 0.05)))
            high_px = max(open_px, close_px) + wick
            low_px = max(0.1, min(open_px, close_px) - wick)
            vol = max(1_000.0, 1_000_000.0 * (1 + min(abs(real_pnl) * 15, 3.0)))
            amount = vol * (open_px + close_px) / 2
            turnover = max(0.005, min(0.35, 0.01 + abs(eff_pnl) * 1.2))
            synth.append(
                {
                    "日期": date,
                    "开盘": open_px,
                    "收盘": close_px,
                    "最高": high_px,
                    "最低": low_px,
                    "成交量": vol,
                    "成交额": amount,
                    "换手率": turnover,
                }
            )
            prev_close = close_px
        return _finalize(pd.DataFrame(synth), turnover_multiplier=100.0)

    def _fallback_from_latest_advice() -> pd.DataFrame:
        payload = _load_latest_advice_payload(ticker)
        if not payload:
            raise RuntimeError(f"历史建议兜底缺失: 未找到 {ticker} 的建议文件")
        latest_market = payload.get("latest_market", {}) or {}
        close_px = max(0.1, _to_float(latest_market.get("close"), 100.0))
        pct_change = _to_float(latest_market.get("pct_change"), 0.0)
        turnover = max(0.01, _to_float(latest_market.get("turnover"), 1.0))
        volume = max(1_000.0, _to_float(latest_market.get("volume"), 1_000_000.0))
        amount = max(1_000.0, volume * close_px)
        as_of_date = str(payload.get("as_of_date") or datetime.now().strftime("%Y-%m-%d"))[:10]
        try:
            dates = pd.bdate_range(end=pd.to_datetime(as_of_date), periods=80)
        except Exception:
            dates = pd.bdate_range(end=pd.Timestamp.today(), periods=80)

        prev_final_close = close_px / max(0.2, 1.0 + pct_change / 100.0)
        price = max(0.1, prev_final_close * 0.96)
        rows = []
        for idx, date in enumerate(dates):
            is_last = idx == len(dates) - 1
            open_px = prev_final_close if is_last else price
            if is_last:
                day_close = close_px
            else:
                wave = ((idx % 9) - 4) * 0.0015
                drift = 0.00035
                day_close = max(0.1, open_px * (1 + drift + wave))
            wick = max(abs(day_close - open_px) * 0.5, open_px * 0.006)
            high_px = max(open_px, day_close) + wick
            low_px = max(0.1, min(open_px, day_close) - wick)
            rows.append(
                {
                    "日期": date.strftime("%Y-%m-%d"),
                    "开盘": open_px,
                    "收盘": day_close,
                    "最高": high_px,
                    "最低": low_px,
                    "成交量": volume,
                    "成交额": amount,
                    "换手率": turnover,
                }
            )
            price = day_close
        return _finalize(pd.DataFrame(rows), turnover_multiplier=1.0)

    prefix = "sh" if ticker.startswith("6") else "sz"
    prefixed = f"{prefix}{ticker}"
    source = "akshare"
    fallback_used = False
    errors: list[str] = []
    df_hist: pd.DataFrame | None = None

    try:
        df_hist = ak.stock_zh_a_hist(symbol=ticker, period="daily", adjust="qfq")
        if df_hist is None or df_hist.empty:
            raise RuntimeError("stock_zh_a_hist returned empty data")
        df_hist = _finalize(df_hist, turnover_multiplier=1.0)
        source = "akshare_eastmoney_hist"
    except Exception as exc:
        errors.append(f"akshare_eastmoney_hist: {exc}")

    if df_hist is None or df_hist.empty:
        try:
            df_hist = ak.stock_zh_a_daily(symbol=prefixed, adjust="qfq")
            df_hist.rename(columns={
                "date": "日期",
                "open": "开盘",
                "high": "最高",
                "low": "最低",
                "close": "收盘",
                "volume": "成交量",
                "amount": "成交额",
                "turnover": "换手率",
            }, inplace=True)
            df_hist = _finalize(df_hist, turnover_multiplier=100.0)
            source = "akshare_sina_daily"
        except Exception as exc:
            errors.append(f"akshare_sina_daily: {exc}")

    if df_hist is None or df_hist.empty:
        print(f"[WARN] 在线行情获取失败({ticker})，启用本地兜底: {'; '.join(errors[-2:])}")
        fallback_used = True
        for fallback_source, fetcher in [
            ("local_backtest_fallback", _fallback_from_local_backtests),
            ("latest_advice_fallback", _fallback_from_latest_advice),
        ]:
            try:
                df_hist = fetcher()
                source = fallback_source
                break
            except Exception as exc:
                errors.append(f"{fallback_source}: {exc}")

    if df_hist is None:
        diagnostics = {
            "source": "marketless_direct_fallback",
            "fallback_used": True,
            "rows": 0,
            "as_of_date": datetime.now().strftime("%Y-%m-%d"),
            "freshness_days": None,
            "min_required_rows": 20,
            "coverage_target_rows": 120,
            "status": "market_unavailable",
            "errors": errors[-6:],
        }
        return pd.DataFrame(), diagnostics

    if df_hist.empty or len(df_hist) < 20:
        raise RuntimeError(f"{ticker} 历史行情不足，无法生成结构化建议")

    as_of_date = str(df_hist.iloc[-1]["日期"])
    diagnostics = {
        "source": source,
        "fallback_used": fallback_used,
        "rows": int(len(df_hist)),
        "as_of_date": as_of_date,
        "freshness_days": _days_since_iso(as_of_date),
        "min_required_rows": 20,
        "coverage_target_rows": 120,
        "status": "fallback" if fallback_used else "ok",
        "errors": errors[:4],
    }
    return df_hist, diagnostics


def fetch_history(ticker: str) -> pd.DataFrame:
    df_hist, _ = fetch_history_with_diagnostics(ticker)
    return df_hist


def _date_int_to_iso(value) -> str:
    try:
        text = str(int(value))
        if len(text) != 8:
            return ""
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    except Exception:
        return ""


SOURCE_LINK_DEFS = {
    "news": {
        "label": "东方财富个股新闻",
        "url": "https://so.eastmoney.com/news/s?keyword={ticker}",
    },
    "market_news": {
        "label": "财新市场要闻",
        "url": "https://www.caixin.com/search/{ticker}.html",
    },
    "report": {
        "label": "东方财富研报",
        "url": "https://data.eastmoney.com/report/{ticker}.html",
    },
    "announcement": {
        "label": "巨潮资讯公告",
        "url": "https://www.cninfo.com.cn/new/fulltextSearch?notautosubmit=&keyWord={ticker}",
    },
    "financial_abstract": {
        "label": "东方财富财务摘要",
        "url": "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={exchange_ticker}#/cwfx",
    },
    "financial_indicator": {
        "label": "东方财富财务指标",
        "url": "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={exchange_ticker}#/cwfx",
    },
    "income_statement": {
        "label": "东方财富利润表",
        "url": "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={exchange_ticker}#/cwfx",
    },
    "balance_sheet": {
        "label": "东方财富资产负债表",
        "url": "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={exchange_ticker}#/cwfx",
    },
    "cash_flow": {
        "label": "东方财富现金流量表",
        "url": "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={exchange_ticker}#/cwfx",
    },
}


def _exchange_ticker_for_link(ticker: str) -> str:
    code = str(ticker or "").strip().zfill(6)
    return ("SH" if code.startswith("6") else "SZ") + code


def _build_source_links(source_types: dict[str, int], ticker: str) -> list[dict]:
    code = str(ticker or "").strip().zfill(6)
    exchange_ticker = _exchange_ticker_for_link(code)
    links = []
    for source, count in sorted(source_types.items(), key=lambda item: (-int(item[1] or 0), str(item[0]))):
        if int(count or 0) <= 0:
            continue
        spec = SOURCE_LINK_DEFS.get(str(source))
        if not spec:
            continue
        url = spec["url"].format(
            ticker=quote(code),
            exchange_ticker=quote(exchange_ticker),
        )
        links.append({"source": source, "label": spec["label"], "url": url, "count": int(count or 0)})
    return links


def _summarize_knowledge(knowledge: list[dict], *, fallback_used: bool, errors: list[str], ticker: str = "") -> dict:
    source_types: dict[str, int] = {}
    latest_date_int = 0
    for item in knowledge or []:
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        source = str(metadata.get("source", "unknown") or "unknown")
        source_types[source] = source_types.get(source, 0) + 1
        try:
            latest_date_int = max(latest_date_int, int(metadata.get("date_int", 0) or 0))
        except Exception:
            pass

    latest_date = _date_int_to_iso(latest_date_int)
    return {
        "source": "external_knowledge",
        "documents": int(len(knowledge or [])),
        "fallback_used": bool(fallback_used),
        "vectorstore_degraded": False,
        "degraded_reason": "",
        "source_types": source_types,
        "source_links": _build_source_links(source_types, ticker),
        "latest_date": latest_date,
        "freshness_days": _days_since_iso(latest_date) if latest_date else None,
        "status": "fallback" if fallback_used else "ok",
        "errors": errors[:3],
    }


def build_rag_with_diagnostics(ticker: str, cutoff_date: str | None = None) -> tuple[SimpleRAG, dict]:
    errors: list[str] = []
    fallback_used = False
    try:
        knowledge = fetch_external_knowledge_for_advice(ticker, cutoff_date=cutoff_date)
    except Exception as exc:
        knowledge = []
        errors.append(str(exc))

    if not knowledge:
        fallback_used = True
        knowledge = [
            {
                "page_content": f"{ticker} 暂无可用新闻研报，建议降低新闻面权重。",
                "metadata": {"date_int": _date_to_int(cutoff_date) if cutoff_date else 20991231, "ticker": ticker, "source": "fallback"},
            }
        ]
    rag_engine = SimpleRAG(data_sources=knowledge)
    diagnostics = _summarize_knowledge(
        knowledge,
        fallback_used=fallback_used,
        errors=errors,
        ticker=ticker,
    )
    if getattr(rag_engine, "degraded_reason", ""):
        diagnostics["vectorstore_degraded"] = True
        diagnostics["degraded_reason"] = str(getattr(rag_engine, "degraded_reason", ""))[:240]
        diagnostics["status"] = "degraded"
    return rag_engine, diagnostics


def build_rag(ticker: str, cutoff_date: str | None = None) -> SimpleRAG:
    rag_engine, _ = build_rag_with_diagnostics(ticker, cutoff_date=cutoff_date)
    return rag_engine


def build_marketless_direct_advice(
    ticker: str,
    market_diagnostics: dict,
    human_comment: str = "",
    human_decision: str | None = None,
) -> dict:
    """Return a conservative advice payload when no usable market history is available."""
    ticker = str(ticker).strip().zfill(6)
    target_date = str((market_diagnostics or {}).get("as_of_date") or datetime.now().strftime("%Y-%m-%d"))
    errors = list((market_diagnostics or {}).get("errors") or [])
    error_text = "；".join(str(x) for x in errors[-4:]) or "在线行情、本地回测与历史建议均不可用"
    requested_action = _normalize_action(human_decision) if human_decision else "HOLD"
    if requested_action in {"BUY", "SELL"}:
        final_action = "HOLD"
        decision_note = f"人工方向为 {requested_action}，但行情数据缺失，系统将执行动作降级为 HOLD。"
    else:
        final_action = "HOLD"
        decision_note = "缺少可验证历史行情，系统不生成方向性买卖信号。"

    technical_reason = (
        f"[规则降级] {ticker} 当前无法取得可用历史行情，无法计算均线、RSI、MACD、波动率与量能指标；"
        "技术面结论降级为中性观望。"
    )
    fundamental_reason = (
        "[规则降级] 未在本次直接兜底中重新拉取新闻研报；若需要完整基本面判断，请在行情源恢复后重新生成建议。"
    )
    if human_comment:
        fundamental_reason = f"{fundamental_reason} 人工备注：{human_comment}"

    technical_features = {
        "version": f"{TECHNICAL_FEATURE_VERSION}_marketless",
        "as_of_date": target_date,
        "lookback_rows": 0,
        "trend": {},
        "momentum": {},
        "risk": {},
        "volume": {},
        "signals": [
            "历史行情不可用，无法计算技术指标。",
            "未验证价格趋势前，不建议生成方向性买卖信号。",
            "如确需交易，应先人工核对盘口、公告与流动性。",
        ],
        "marketless_fallback": True,
    }
    technical_case = {
        "agent": "技术资金综合分析师",
        "sentiment": "neutral",
        "confidence": 0.18,
        "reasoning": technical_reason,
        "source_reports": [
            {
                "agent": "行情数据诊断",
                "sentiment": "neutral",
                "confidence": 0.1,
                "reasoning": error_text,
                "_parse_ok": True,
            }
        ],
    }
    fundamental_case = {
        "agent": "基本面新闻综合分析师",
        "sentiment": "neutral",
        "confidence": 0.2,
        "reasoning": fundamental_reason,
        "source_reports": [],
    }
    referee = {
        "decision": "HOLD",
        "bull_score": 0.0,
        "bear_score": 0.0,
        "trend_strength": 0.0,
        "confidence": 0.2,
        "reason": f"行情证据链断裂，无法支持 {requested_action} 的方向性判断；默认保持观望。",
        "debate_trace": [
            {
                "round": 1,
                "judge": {
                    "decision": "HOLD",
                    "sentiment": "neutral",
                    "confidence": 0.2,
                    "reasoning": "行情数据不可用，技术面与基本面都无法给出可验证优势。",
                },
            }
        ],
    }
    risk = {
        "decision": final_action,
        "action": final_action,
        "final_action": final_action,
        "execution_action": final_action,
        "position_percent": 0.0,
        "reason": f"{decision_note} 风控要求先恢复行情数据或人工核验后再考虑交易。",
    }
    rag_diagnostics = {
        "source": "marketless_direct_fallback",
        "documents": 0,
        "fallback_used": True,
        "vectorstore_degraded": False,
        "degraded_reason": "",
        "source_types": {"fallback": 1},
        "latest_date": target_date,
        "freshness_days": None,
        "status": "marketless",
        "errors": [],
    }
    data_quality = assess_data_quality(
        market_diagnostics=market_diagnostics,
        rag_diagnostics=rag_diagnostics,
        technical_case=technical_case,
        fundamental_case=fundamental_case,
    )
    data_quality["score"] = min(float(data_quality.get("score", 0.0) or 0.0), 0.25)
    data_quality["level"] = "low"
    data_quality["note"] = "行情数据不可用，本次仅生成保守观望建议；请恢复行情源后重新生成完整报告。"
    data_quality.setdefault("diagnostics", [])
    data_quality["diagnostics"] = [
        "行情数据不可用，使用无行情直接兜底",
        *[item for item in data_quality["diagnostics"] if item != "行情数据不可用，使用无行情直接兜底"],
    ][:6]

    recommendation = {
        "action": final_action,
        "execution_action": final_action,
        "position_percent": 0.0,
        "confidence": 0.2,
        "reason": (
            f"未取得 {ticker} 的可用历史行情，系统无法验证趋势、波动与量能；"
            "因此本次建议为 HOLD/0% 仓位。请检查股票代码、行情源或稍后重试。"
        ),
        "next_action": "恢复行情数据后重新生成完整建议。",
    }
    recommendation = calibrate_recommendation_by_quality(recommendation, risk, data_quality)
    advice = {
        "ticker": ticker,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of_date": target_date,
        "latest_market": {
            "close": None,
            "pct_change": None,
            "turnover": None,
            "volume": None,
            "source": "marketless_direct_fallback",
        },
        "technical_features": technical_features,
        "recommendation": recommendation,
        "analyst_cases": {
            "technical_flow": technical_case,
            "fundamental_news": fundamental_case,
        },
        "referee": referee,
        "risk": risk,
        "data_quality": data_quality,
        "warnings": [
            "本建议未使用历史行情K线，仅为行情不可用时的保守兜底结果。",
            "请勿把本次 HOLD 视为完整技术面/基本面结论。",
        ],
        "cache_hit": False,
        "marketless_fallback": True,
    }
    multi_period_advice = {
        "short_term": {
            "horizon": "1-3 trading days",
            "action": "HOLD",
            "confidence": 0.2,
            "reason": "短线缺少行情验证，不追涨不杀跌，等待数据恢复。",
        },
        "swing_term": {
            "horizon": "2-4 weeks",
            "action": "HOLD",
            "confidence": 0.2,
            "reason": "中线缺少趋势、波动和量能证据，不建立方向性仓位。",
        },
        "risk_plan": {
            "stop_loss": "未建立新仓；已有持仓请以人工确认的关键支撑/成本线为准。",
            "take_profit": "未建立新仓；已有盈利持仓可按既有交易纪律分批锁定收益。",
            "invalid_conditions": [
                "行情数据仍不可用",
                "无法核验最新价格与流动性",
                "公告或新闻出现重大不确定性但未完成复核",
            ],
        },
    }
    advice["multi_period_advice"] = multi_period_advice
    advice.update(multi_period_advice)
    stability = _compute_stability(advice)
    advice.update(stability)
    advice["stability_score"] = min(float(advice.get("stability_score", 0.0) or 0.0), 0.35)
    advice["stability_level"] = "low"
    advice["stability_note"] = "行情缺失兜底建议，仅供记录与排障参考。"
    advice["recommendation"]["stability_note"] = advice["stability_note"]
    _emit_progress(
        "finalize",
        96,
        "无行情保守兜底建议已生成：HOLD / 0% 仓位",
        title="最终建议",
        status="warning",
        payload={
            "action": "HOLD",
            "position_percent": 0.0,
            "confidence": 0.2,
            "stability_level": "low",
        },
        partial_result={
            "recommendation": advice.get("recommendation", {}),
            "data_quality": advice.get("data_quality", {}),
            "multi_period_advice": advice.get("multi_period_advice", {}),
        },
    )
    return advice


def generate_advice(
    ticker: str,
    debate_depth: int,
    human_comment: str = "",
    human_decision: str | None = None,
    skip_strategy_rules: bool = False,
) -> dict:
    ticker = str(ticker).strip().zfill(6)
    _emit_progress("start", 3, f"开始生成 {ticker} 结构化投资建议", title="任务启动", status="running")
    memory_bank = MemoryBank()
    _emit_progress("market_data", 8, "正在获取历史行情与本地兜底数据", title="行情获取", status="running")
    df_hist, market_diagnostics = fetch_history_with_diagnostics(ticker)
    if df_hist.empty:
        _emit_progress(
            "market_data",
            18,
            "行情获取失败，进入无行情保守兜底",
            title="行情获取",
            status="warning",
            payload=market_diagnostics,
            partial_result={"data_quality": {"level": "low"}, "latest_market": {"source": "marketless_direct_fallback"}},
        )
        return build_marketless_direct_advice(
            ticker=ticker,
            market_diagnostics=market_diagnostics,
            human_comment=human_comment,
            human_decision=human_decision,
        )
    target_date = str(df_hist.iloc[-1]["日期"])
    _emit_progress(
        "market_data",
        16,
        f"行情获取完成：{market_diagnostics.get('source')}，{market_diagnostics.get('rows')} 条，最新 {target_date}",
        title="行情获取",
        payload=market_diagnostics,
        partial_result={
            "latest_market": {
                "as_of_date": target_date,
                "source": market_diagnostics.get("source"),
                "rows": market_diagnostics.get("rows"),
            }
        },
    )
    if not human_comment and not human_decision:
        cached = _load_stable_cache(ticker, target_date)
        if cached:
            cache_errors = _validate_advice_payload(cached)
            if (
                not cache_errors
                and cached.get("data_quality")
                and _is_stable_cacheable_advice(cached)
                and _cache_matches_current_dl_backend(cached)
            ):
                cached = dict(cached)
                cached["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cached["cache_hit"] = True
                _emit_progress(
                    "cache",
                    100,
                    "命中稳定缓存，直接返回最新可复用建议",
                    title="稳定缓存",
                    payload={"as_of_date": target_date},
                    partial_result={"recommendation": cached.get("recommendation", {})},
                )
                return cached
    technical_feature_bundle = build_technical_feature_bundle(df_hist)
    latest = df_hist.iloc[-1]
    tech_values = technical_feature_bundle.get("feature_values", {})
    _emit_progress(
        "technical_features",
        24,
        "技术特征计算完成",
        title="技术特征",
        payload={
            "as_of_date": tech_values.get("as_of_date"),
            "lookback_rows": tech_values.get("lookback_rows"),
            "signals": (tech_values.get("signals") or [])[:4],
        },
        partial_result={"technical_features": tech_values},
    )

    _emit_progress("rag", 28, "正在检索新闻、研报、公告与财务证据", title="RAG证据检索", status="running")
    rag_engine, rag_diagnostics = build_rag_with_diagnostics(ticker, cutoff_date=target_date)
    _emit_progress(
        "rag",
        38,
        f"RAG证据检索完成：{rag_diagnostics.get('documents', 0)} 份文档",
        title="RAG证据检索",
        payload=rag_diagnostics,
        partial_result={"rag": rag_diagnostics},
    )
    dl_engine = None
    if str(os.getenv("ENABLE_DL_ANALYST", "1")).strip() != "0":
        from dl.predictor import DLEngine

        dl_engine = DLEngine()
    technical_flow_analyst = TechnicalFlowAnalyst(name="技术资金综合分析师", dl_engine=dl_engine)
    fundamental_news_analyst = FundamentalNewsAnalyst(name="基本面新闻综合分析师", rag_engine=rag_engine)
    referee = GameReferee(name="无情裁判官", memory_bank=memory_bank)
    risk_manager = RiskManager(name="风控大脑", memory_bank=memory_bank)
    trader = TraderAgent(name="结构化建议执行器")

    _emit_progress("analysts", 42, "技术资金与基本面新闻分析师开始并行研判", title="分析师研判", status="running")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_technical = executor.submit(
            technical_flow_analyst.step,
            ticker,
            features=technical_feature_bundle,
            target_date=target_date,
        )
        future_fundamental = executor.submit(
            fundamental_news_analyst.step,
            ticker,
            target_date=target_date,
        )
        technical_case = future_technical.result()
        fundamental_case = future_fundamental.result()
    _emit_progress(
        "technical_analysis",
        55,
        f"技术资金分析完成：{technical_case.get('sentiment', 'neutral')} / 置信度 {technical_case.get('confidence', '-')}",
        title="技术资金分析",
        payload={
            "sentiment": technical_case.get("sentiment"),
            "confidence": technical_case.get("confidence"),
            "reasoning": str(technical_case.get("reasoning", ""))[:220],
        },
        partial_result={"technical_case": technical_case},
    )
    _emit_progress(
        "fundamental_analysis",
        64,
        f"基本面新闻分析完成：{fundamental_case.get('sentiment', 'neutral')} / 置信度 {fundamental_case.get('confidence', '-')}",
        title="基本面新闻分析",
        payload={
            "sentiment": fundamental_case.get("sentiment"),
            "confidence": fundamental_case.get("confidence"),
            "reasoning": str(fundamental_case.get("reasoning", ""))[:220],
        },
        partial_result={"fundamental_case": fundamental_case},
    )
    _emit_progress("debate", 68, "裁判开始融合多空观点与人工输入", title="裁判博弈", status="running")
    referee_decision = referee.step_agent_game(
        analyst_a=technical_flow_analyst,
        analyst_b=fundamental_news_analyst,
        case_a=technical_case,
        case_b=fundamental_case,
        ticker=ticker,
        max_depth=debate_depth,
        human_comment=human_comment,
        human_decision=human_decision,
    )
    _emit_progress(
        "debate",
        76,
        f"裁判博弈完成：{referee_decision.get('decision', 'HOLD')} / 置信度 {referee_decision.get('confidence', '-')}",
        title="裁判博弈",
        payload={
            "decision": referee_decision.get("decision"),
            "confidence": referee_decision.get("confidence"),
            "bull_score": referee_decision.get("bull_score"),
            "bear_score": referee_decision.get("bear_score"),
            "reason": str(referee_decision.get("reason", ""))[:220],
        },
        partial_result={"referee": referee_decision},
    )
    _emit_progress("risk", 80, "风控开始约束最终动作与仓位", title="风控执行", status="running")
    final_instruction = risk_manager.step(ticker, referee_decision)
    execution_action = trader.step(final_instruction)
    _emit_progress(
        "risk",
        86,
        f"风控完成：{final_instruction.get('decision', 'HOLD')}，仓位 {final_instruction.get('position_percent', 0)}%",
        title="风控执行",
        payload={
            "decision": final_instruction.get("decision"),
            "position_percent": final_instruction.get("position_percent"),
            "execution_action": execution_action,
            "reason": str(final_instruction.get("reason", ""))[:220],
        },
        partial_result={"risk": final_instruction},
    )
    data_quality = assess_data_quality(
        market_diagnostics=market_diagnostics,
        rag_diagnostics=rag_diagnostics,
        technical_case=technical_case,
        fundamental_case=fundamental_case,
    )
    _emit_progress(
        "data_quality",
        90,
        f"数据质量评估完成：{data_quality.get('level')} / {data_quality.get('score')}",
        title="数据质量",
        payload={
            "score": data_quality.get("score"),
            "level": data_quality.get("level"),
            "diagnostics": data_quality.get("diagnostics", []),
        },
        partial_result={"data_quality": data_quality},
    )

    advice = {
        "ticker": ticker,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of_date": target_date,
        "latest_market": {
            "close": float(latest["收盘"]),
            "pct_change": float(latest["涨跌幅"]),
            "turnover": float(latest["换手率"]),
            "volume": float(latest["成交量"]),
        },
        "technical_features": technical_feature_bundle["feature_values"],
        "recommendation": {
            "action": final_instruction["decision"],
            "execution_action": execution_action,
            "position_percent": final_instruction.get("position_percent", 0.0),
            "confidence": referee_decision.get("confidence", 0.5),
            "reason": final_instruction.get("reason", referee_decision.get("reason", "")),
            "next_action": referee_decision.get("next_action", ""),
        },
        "analyst_cases": {
            "technical_flow": technical_case,
            "fundamental_news": fundamental_case,
        },
        "referee": {
            "decision": referee_decision.get("decision"),
            "bull_score": referee_decision.get("bull_score"),
            "bear_score": referee_decision.get("bear_score"),
            "trend_strength": referee_decision.get("trend_strength"),
            "debate_trace": referee_decision.get("debate_trace", []),
        },
        "risk": final_instruction,
        "data_quality": data_quality,
    }
    advice["recommendation"] = calibrate_recommendation_by_quality(
        advice["recommendation"],
        advice["risk"],
        data_quality,
    )
    if skip_strategy_rules:
        advice["strategy_rules"] = {
            "applied": False,
            "candidate_count": 0,
            "applied_rules": [],
            "user_rule_preference": "disabled",
        }
    else:
        advice = apply_strategy_rules_to_advice(advice)
    multi_period_advice = build_multi_period_advice(
        recommendation=advice["recommendation"],
        risk=advice["risk"],
        referee=advice["referee"],
        analyst_cases=advice["analyst_cases"],
        technical_features=advice["technical_features"],
        data_quality=advice["data_quality"],
    )
    advice["multi_period_advice"] = multi_period_advice
    advice.update(multi_period_advice)
    advice["cache_hit"] = False
    stability = _compute_stability(advice)
    advice.update(stability)
    advice["recommendation"]["stability_note"] = stability["stability_note"]
    _emit_progress(
        "finalize",
        96,
        f"多周期建议与稳定度评估完成：{advice['recommendation'].get('action')} / {advice.get('stability_level')}",
        title="最终建议",
        payload={
            "action": advice["recommendation"].get("action"),
            "position_percent": advice["recommendation"].get("position_percent"),
            "confidence": advice["recommendation"].get("confidence"),
            "stability_level": advice.get("stability_level"),
        },
        partial_result={"recommendation": advice.get("recommendation", {}), "multi_period_advice": multi_period_advice},
    )
    return advice


def generate_advice_with_retry(
    ticker: str,
    debate_depth: int,
    human_comment: str = "",
    human_decision: str | None = None,
    skip_strategy_rules: bool = False,
    max_retries: int = MAX_GENERATION_RETRY,
) -> dict:
    last_errors: list[str] = []
    for attempt in range(1, max_retries + 1):
        advice = generate_advice(
            ticker=ticker,
            debate_depth=debate_depth,
            human_comment=human_comment,
            human_decision=human_decision,
            skip_strategy_rules=skip_strategy_rules,
        )
        errors = _validate_advice_payload(advice)
        if not errors:
            if not advice.get("cache_hit"):
                _write_stable_cache(advice)
            return advice
        last_errors = errors
        print(f"[WARN] 第 {attempt}/{max_retries} 次生成未通过关键字段校验: {'; '.join(errors)}")
    raise RuntimeError(f"关键字段校验失败，拒绝落库: {'; '.join(last_errors)}")


def main() -> None:
    args = _parse_args()
    advice = generate_advice_with_retry(
        ticker=args.ticker,
        debate_depth=args.debate_depth,
        human_comment=args.human_comment,
        human_decision=args.human_decision,
        skip_strategy_rules=args.skip_strategy_rules,
    )
    output = json.dumps(advice, ensure_ascii=False, indent=2)
    print(output)

    if args.output_path:
        output_path = args.output_path
    else:
        output_dir = os.path.join(BASE_DIR, "data", "investment_advice")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{advice['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    _emit_progress(
        "output",
        99,
        "结构化建议 JSON 已保存，前端即将渲染结果",
        title="结果保存",
        payload={"output_path": output_path},
    )
    print(f"\n结构化投资建议已保存: {output_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    if str(os.getenv("ADVICE_SCRIPT_FORCE_EXIT", "1")).strip() != "0":
        os._exit(0)


if __name__ == "__main__":
    main()
