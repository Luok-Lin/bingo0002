from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from .config import ADVICE_DIR, ADVICE_SETTLEMENT_PATH, BACKTEST_SUMMARY_DIR, INDEX_SUMMARY_PATH, REFLECTIONS_PATH
from .storage import read_json, write_json


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _collect_latest_advice() -> dict[str, dict]:
    latest: dict[str, tuple[str, dict]] = {}
    if not os.path.isdir(ADVICE_DIR):
        return {}

    for name in os.listdir(ADVICE_DIR):
        if not name.endswith(".json"):
            continue
        if name in {"top10_screening_latest.json", "rescreen_top3_latest.json", "post_tune_validation_summary.json"}:
            continue
        path = os.path.join(ADVICE_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        ticker = str(payload.get("ticker", "")).zfill(6)
        if not ticker:
            continue
        generated_at = str(payload.get("generated_at", ""))
        # Fallback to file mtime if generated_at missing.
        if not generated_at:
            generated_at = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
        current = latest.get(ticker)
        if current is None or generated_at >= current[0]:
            latest[ticker] = (generated_at, payload)
    return {k: v[1] for k, v in latest.items()}


def _collect_advice_total_count() -> int:
    """
    Count all valid advice snapshots in ADVICE_DIR.
    Unlike latest_advice_count (unique tickers), this keeps historical runs.
    """
    if not os.path.isdir(ADVICE_DIR):
        return 0
    total = 0
    for name in os.listdir(ADVICE_DIR):
        if not name.endswith(".json"):
            continue
        if name in {"top10_screening_latest.json", "rescreen_top3_latest.json", "post_tune_validation_summary.json"}:
            continue
        path = os.path.join(ADVICE_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        ticker = str(payload.get("ticker", "")).zfill(6)
        rec = payload.get("recommendation", {})
        if not ticker or not isinstance(rec, dict):
            continue
        total += 1
    return total


def _collect_latest_backtest_runs(limit: int = 20) -> list[dict]:
    runs = []
    if not os.path.isdir(BACKTEST_SUMMARY_DIR):
        return runs
    for name in os.listdir(BACKTEST_SUMMARY_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(BACKTEST_SUMMARY_DIR, name)
        runs.append((os.path.getmtime(path), path))
    runs.sort(reverse=True)

    out: list[dict] = []
    for _, path in runs[:limit]:
        payload = read_json(path, {})
        if payload:
            out.append(payload)
    return out


def _reflection_stats() -> dict:
    reflections = read_json(REFLECTIONS_PATH, [])
    if not isinstance(reflections, list):
        return {}

    action_counter = Counter()
    rewards: list[float] = []
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for row in reflections:
        action = str(row.get("decision", "UNKNOWN")).upper()
        action_counter[action] += 1
        rewards.append(_safe_float(row.get("reward_score"), 0.0))
        ticker = str(row.get("ticker", "UNKNOWN")).zfill(6)
        by_ticker[ticker].append(row)

    ticker_win_rates: list[dict] = []
    for ticker, rows in by_ticker.items():
        trade_rows = [r for r in rows if str(r.get("decision", "")).upper() in {"BUY", "SELL"}]
        if not trade_rows:
            continue
        wins = 0
        for row in trade_rows:
            wins += 1 if _safe_float(row.get("pnl_percent"), 0.0) > 0 else 0
        ticker_win_rates.append(
            {
                "ticker": ticker,
                "trades": len(trade_rows),
                "win_rate": round(wins / len(trade_rows), 4),
                "avg_reward": round(sum(_safe_float(r.get("reward_score"), 0.0) for r in trade_rows) / len(trade_rows), 4),
            }
        )
    ticker_win_rates.sort(key=lambda x: x["win_rate"], reverse=True)

    reward_avg = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
    reward_neg_ratio = round(sum(1 for x in rewards if x < 0) / len(rewards), 4) if rewards else 0.0
    return {
        "rows": len(reflections),
        "action_distribution": dict(action_counter),
        "reward_avg": reward_avg,
        "reward_neg_ratio": reward_neg_ratio,
        "top_ticker_win_rates": ticker_win_rates[:10],
    }


def _advice_settlement_stats() -> dict:
    state = read_json(ADVICE_SETTLEMENT_PATH, {})
    if not isinstance(state, dict):
        state = {}
    settled = state.get("settled_keys", [])
    settled_set = set(str(x) for x in settled) if isinstance(settled, list) else set()
    total_files = _collect_advice_total_count()
    settled_count = len(settled_set)
    pending_count = max(0, total_files - settled_count)
    return {
        "settled_count": settled_count,
        "pending_count": pending_count,
        "updated_at": state.get("updated_at", ""),
    }


def build_dashboard_summary() -> dict:
    latest_advice = _collect_latest_advice()
    advice_total_count = _collect_advice_total_count()
    advice_actions = Counter()
    advice_rows: list[dict] = []
    for ticker, payload in latest_advice.items():
        rec = payload.get("recommendation", {})
        action = str(rec.get("action", "HOLD")).upper()
        advice_actions[action] += 1
        advice_rows.append(
            {
                "ticker": ticker,
                "generated_at": payload.get("generated_at"),
                "action": action,
                "position_percent": _safe_float(rec.get("position_percent"), 0.0),
                "confidence": _safe_float(rec.get("confidence"), 0.0),
                "reason": rec.get("reason", ""),
            }
        )
    advice_rows.sort(key=lambda x: x["generated_at"] or "", reverse=True)

    summary = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "advice_total_count": advice_total_count,
        "latest_advice_count": len(latest_advice),
        "advice_action_distribution": dict(advice_actions),
        "latest_advice": advice_rows[:50],
        "recent_backtest_runs": _collect_latest_backtest_runs(),
        "reflection_stats": _reflection_stats(),
        "advice_settlement_stats": _advice_settlement_stats(),
    }
    write_json(INDEX_SUMMARY_PATH, summary)
    return summary

