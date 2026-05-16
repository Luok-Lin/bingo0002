from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

import akshare as ak

from rl.reward import compute_trade_reward

from .config import (
    ADVICE_DIR,
    ADVICE_SETTLEMENT_PATH,
    BACKTEST_SUMMARY_DIR,
    EVOLUTION_HISTORY_PATH,
    REFLECTIONS_PATH,
    TOP_HOLDINGS_CSV,
    WATCHLIST_PATH,
)
from .indexer import build_dashboard_summary
from .storage import read_json, write_json


def _run_command(cmd: list[str], cwd: str) -> dict:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-5000:],
        "stderr": proc.stderr[-5000:],
    }


def _latest_file_from_dir(path: str, prefix: str = "", suffix: str = ".json") -> str | None:
    if not os.path.isdir(path):
        return None
    candidates = []
    for name in os.listdir(path):
        if prefix and not name.startswith(prefix):
            continue
        if suffix and not name.endswith(suffix):
            continue
        full = os.path.join(path, name)
        candidates.append((os.path.getmtime(full), full))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _load_top_tickers(csv_path: str, top_n: int) -> list[str]:
    tickers: list[str] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows[:top_n]:
        code = str(row.get("股票代码", "")).strip()
        if code:
            tickers.append(code.zfill(6))
    return tickers


def run_initial_training(
    base_dir: str,
    top_n: int,
    days: int,
    debate_depth: int,
    skip_auto_tune: bool,
    csv_path: str | None = None,
) -> dict:
    source_csv = str(csv_path or TOP_HOLDINGS_CSV)
    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"训练CSV不存在: {source_csv}")
    cmd = [
        sys.executable,
        os.path.join(base_dir, "scripts", "train_top_holdings.py"),
        "--csv-path",
        source_csv,
        "--top-n",
        str(top_n),
        "--days",
        str(days),
        "--debate-depth",
        str(debate_depth),
    ]
    if skip_auto_tune:
        cmd.append("--skip-auto-tune")

    result = _run_command(cmd, cwd=base_dir)
    summary_path = _latest_file_from_dir(os.path.join(base_dir, "data", "monitoring", "portfolio_training"), prefix="top_holdings_")
    write_json(
        WATCHLIST_PATH,
        {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_csv": source_csv,
            "tickers": _load_top_tickers(source_csv, top_n),
            "top_n": top_n,
        },
    )
    dashboard = build_dashboard_summary()
    return {
        "command": cmd,
        "run_result": result,
        "portfolio_summary_path": summary_path,
        "dashboard_updated_at": dashboard.get("updated_at"),
    }


def run_investment_advice(
    base_dir: str,
    ticker: str,
    debate_depth: int = 2,
    human_comment: str = "",
    human_decision: str | None = None,
) -> dict:
    ticker = str(ticker).strip().zfill(6)
    output_dir = os.path.join(base_dir, "data", "investment_advice")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    cmd = [
        sys.executable,
        os.path.join(base_dir, "scripts", "investment_advice.py"),
        ticker,
        "--debate-depth",
        str(debate_depth),
        "--output-path",
        output_path,
    ]
    if human_comment:
        cmd.extend(["--human-comment", human_comment])
    if human_decision:
        cmd.extend(["--human-decision", human_decision])
    run_result = _run_command(cmd, cwd=base_dir)
    advice = read_json(output_path, {}) if run_result.get("returncode") == 0 else {}
    dashboard = build_dashboard_summary()
    return {
        "command": cmd,
        "run_result": run_result,
        "output_path": output_path,
        "advice": advice,
        "dashboard_updated_at": dashboard.get("updated_at"),
    }


def get_latest_advice_for_ticker(ticker: str) -> dict:
    ticker = str(ticker).strip().zfill(6)
    if not os.path.isdir(ADVICE_DIR):
        return {}
    candidates = []
    for name in os.listdir(ADVICE_DIR):
        if not (name.startswith(f"{ticker}_") and name.endswith(".json")):
            continue
        full = os.path.join(ADVICE_DIR, name)
        candidates.append((os.path.getmtime(full), full))
    if not candidates:
        return {}
    candidates.sort(reverse=True)
    return read_json(candidates[0][1], {})


def list_recent_backtest_summaries(limit: int = 20) -> list[dict]:
    if not os.path.isdir(BACKTEST_SUMMARY_DIR):
        return []
    candidates = []
    for name in os.listdir(BACKTEST_SUMMARY_DIR):
        if name.endswith(".json"):
            path = os.path.join(BACKTEST_SUMMARY_DIR, name)
            candidates.append((os.path.getmtime(path), path))
    candidates.sort(reverse=True)
    out = []
    for _, path in candidates[:limit]:
        payload = read_json(path, {})
        if payload:
            out.append(payload)
    return out


def run_daily_evolution(
    base_dir: str,
    tickers: list[str] | None = None,
    debate_depth: int = 2,
    mode: str = "backtest_update",
) -> dict:
    if not tickers:
        watchlist = read_json(WATCHLIST_PATH, {})
        tickers = [str(t).zfill(6) for t in watchlist.get("tickers", [])]
        if not tickers:
            tickers = _load_top_tickers(TOP_HOLDINGS_CSV, 10)

    tickers = [str(t).zfill(6) for t in tickers]
    records: list[dict[str, Any]] = []
    for ticker in tickers:
        if mode == "advice_only":
            result = run_investment_advice(base_dir=base_dir, ticker=ticker, debate_depth=debate_depth)
        else:
            cmd = [
                sys.executable,
                os.path.join(base_dir, "main.py"),
                ticker,
                "1",
                "--debate-depth",
                str(debate_depth),
                "--no-train",
            ]
            run_result = _run_command(cmd, cwd=base_dir)
            result = {"command": cmd, "run_result": run_result}
        records.append(
            {
                "ticker": ticker,
                "returncode": result.get("run_result", {}).get("returncode"),
                "stdout_tail": result.get("run_result", {}).get("stdout", "")[-800:],
                "stderr_tail": result.get("run_result", {}).get("stderr", "")[-800:],
            }
        )

    history = read_json(EVOLUTION_HISTORY_PATH, [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "debate_depth": debate_depth,
            "tickers": tickers,
            "records": records,
        }
    )
    write_json(EVOLUTION_HISTORY_PATH, history[-100:])
    dashboard = build_dashboard_summary()
    return {
        "mode": mode,
        "tickers": tickers,
        "records": records,
        "dashboard_updated_at": dashboard.get("updated_at"),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_action(value: str) -> str:
    token = str(value or "").upper().strip()
    if "BUY" in token:
        return "BUY"
    if "SELL" in token:
        return "SELL"
    return "HOLD"


def _extract_position_fraction(payload: dict) -> float:
    rec = payload.get("recommendation", {}) or {}
    risk = payload.get("risk", {}) or {}
    raw = rec.get("position_percent")
    if raw is None:
        raw = risk.get("position_percent", 0.0)
    position_pct = max(0.0, min(100.0, _safe_float(raw, 0.0)))
    return position_pct / 100.0


def _load_price_map_for_ticker(ticker: str) -> dict[str, float]:
    prefix = "sh" if str(ticker).startswith("6") else "sz"
    symbol = f"{prefix}{str(ticker).zfill(6)}"
    df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
    if df is None or df.empty:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        date = str(row.get("date", "")).split(" ")[0]
        close = _safe_float(row.get("close"), default=float("nan"))
        if date and close == close:
            out[date] = close
    return out


def _load_advice_files() -> list[tuple[str, dict]]:
    if not os.path.isdir(ADVICE_DIR):
        return []
    rows: list[tuple[str, dict]] = []
    for name in os.listdir(ADVICE_DIR):
        if not name.endswith(".json"):
            continue
        if name in {"top10_screening_latest.json", "rescreen_top3_latest.json", "post_tune_validation_summary.json"}:
            continue
        path = os.path.join(ADVICE_DIR, name)
        payload = read_json(path, {})
        if not isinstance(payload, dict):
            continue
        ticker = str(payload.get("ticker", "")).zfill(6)
        if not ticker:
            continue
        rows.append((path, payload))
    rows.sort(key=lambda x: os.path.getmtime(x[0]))
    return rows


def _load_settlement_state() -> dict:
    state = read_json(ADVICE_SETTLEMENT_PATH, {})
    if not isinstance(state, dict):
        state = {}
    settled = state.get("settled_keys", [])
    if not isinstance(settled, list):
        settled = []
    return {
        "updated_at": state.get("updated_at", ""),
        "settled_keys": [str(x) for x in settled if str(x).strip()],
        "records": state.get("records", []) if isinstance(state.get("records"), list) else [],
    }


def _save_settlement_state(state: dict) -> None:
    state = dict(state)
    state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["settled_keys"] = list(dict.fromkeys(state.get("settled_keys", [])))
    state["records"] = (state.get("records") or [])[-5000:]
    write_json(ADVICE_SETTLEMENT_PATH, state)


def _next_trade_date(dates_sorted: list[str], as_of_date: str) -> str:
    for d in dates_sorted:
        if d > as_of_date:
            return d
    return ""


def settle_advice_experience(base_dir: str, max_items: int = 2000) -> dict:
    _ = base_dir  # 与其他 service 保持签名一致，当前逻辑使用全局路径即可
    state = _load_settlement_state()
    settled_keys = set(state.get("settled_keys", []))

    reflections = read_json(REFLECTIONS_PATH, [])
    if not isinstance(reflections, list):
        reflections = []
    ticker_hist_pnls: dict[str, list[float]] = defaultdict(list)
    ticker_recent_actions: dict[str, list[str]] = defaultdict(list)
    for row in reflections:
        ticker = str(row.get("ticker", "")).zfill(6)
        if not ticker:
            continue
        if str(row.get("decision", "")).upper() in {"BUY", "SELL"}:
            ticker_hist_pnls[ticker].append(_safe_float(row.get("pnl_percent"), 0.0))
        ticker_recent_actions[ticker].append(str(row.get("decision", "HOLD")).upper())

    advice_rows = _load_advice_files()
    advice_rows = advice_rows[-max_items:]
    price_cache: dict[str, dict[str, float]] = {}

    settled_count = 0
    pending_count = 0
    skipped_count = 0
    new_records: list[dict] = []

    for file_path, payload in advice_rows:
        file_key = os.path.basename(file_path)
        if file_key in settled_keys:
            continue
        ticker = str(payload.get("ticker", "")).zfill(6)
        action = _normalize_action((payload.get("recommendation", {}) or {}).get("action", "HOLD"))
        as_of_date = str(payload.get("as_of_date", "")).strip()
        if not ticker or not as_of_date:
            skipped_count += 1
            continue

        if ticker not in price_cache:
            try:
                price_cache[ticker] = _load_price_map_for_ticker(ticker)
            except Exception:
                price_cache[ticker] = {}
        price_map = price_cache[ticker]
        dates_sorted = sorted(price_map.keys())
        next_date = _next_trade_date(dates_sorted, as_of_date)
        if not next_date:
            pending_count += 1
            continue

        close_t = _safe_float(price_map.get(as_of_date), default=float("nan"))
        close_t1 = _safe_float(price_map.get(next_date), default=float("nan"))
        if close_t != close_t or close_t1 != close_t1 or close_t <= 0:
            pending_count += 1
            continue

        market_move_percent = (close_t1 - close_t) / close_t * 100.0
        position = _extract_position_fraction(payload)
        actual_pnl_percent = (
            market_move_percent * position
            if action == "BUY"
            else ((-market_move_percent) * position if action == "SELL" else 0.0)
        )
        reward = compute_trade_reward(
            action=action,
            actual_pnl_percent=actual_pnl_percent,
            market_move_percent=market_move_percent,
            historical_pnls=ticker_hist_pnls[ticker],
            recent_actions=ticker_recent_actions[ticker][-20:],
            position=position,
        )
        record = {
            "ticker": ticker,
            "decision": action,
            "pnl_percent": round(actual_pnl_percent, 4),
            "market_move_percent": round(market_move_percent, 4),
            "position": round(position, 4),
            "reward_score": round(reward.reward, 4),
            "reward_text": (
                f"reward={reward.reward:.3f}, return={reward.return_component:.3f}, "
                f"risk_penalty={reward.volatility_penalty + reward.drawdown_penalty + reward.loss_streak_penalty:.3f}, "
                f"behavior_penalty={reward.turnover_penalty + reward.exposure_penalty + reward.hold_penalty + reward.hold_streak_penalty:.3f}"
            ),
            "reward_stats": reward.to_dict(),
            "reflection_text": f"建议兑现: {as_of_date} -> {next_date}，动作={action}，收益={actual_pnl_percent:.2f}%。",
            "math_stats": f"T+1收益结算，标的波动={market_move_percent:.2f}%，仓位={position*100:.1f}%。",
            "source": "advice_settlement",
            "advice_file": file_key,
            "as_of_date": as_of_date,
            "settled_date": next_date,
            "settled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        reflections.append(record)
        new_records.append(record)
        ticker_hist_pnls[ticker].append(actual_pnl_percent)
        ticker_recent_actions[ticker].append(action)
        settled_keys.add(file_key)
        settled_count += 1

    if new_records:
        write_json(REFLECTIONS_PATH, reflections)
    state["settled_keys"] = list(settled_keys)
    state["records"] = (state.get("records") or []) + new_records
    _save_settlement_state(state)
    dashboard = build_dashboard_summary()
    return {
        "status": "ok",
        "settled_count": settled_count,
        "pending_count": pending_count,
        "skipped_count": skipped_count,
        "dashboard_updated_at": dashboard.get("updated_at"),
    }

