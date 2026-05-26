from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from .config import ADVICE_DIR, ADVICE_SETTLEMENT_PATH, BACKTEST_SUMMARY_DIR, INDEX_SUMMARY_PATH, REFLECTIONS_PATH
from .storage import read_json, write_json

_SKIPPED_ADVICE_FILES = {"top10_screening_latest.json", "rescreen_top3_latest.json", "post_tune_validation_summary.json"}
_ADVICE_EVALUATION_HORIZONS = (1, 3, 5, 10, 20)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_ticker(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:].zfill(6)
    if raw.isdigit():
        return raw.zfill(6)
    return ""


def _iter_advice_payloads():
    if not os.path.isdir(ADVICE_DIR):
        return

    for name in os.listdir(ADVICE_DIR):
        if not name.endswith(".json"):
            continue
        if name in _SKIPPED_ADVICE_FILES:
            continue
        path = os.path.join(ADVICE_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        yield path, payload


def _collect_advice_snapshot() -> tuple[dict[str, dict], int]:
    latest: dict[str, tuple[str, dict]] = {}
    total = 0

    for path, payload in _iter_advice_payloads():
        ticker = _normalize_ticker(payload.get("ticker", ""))
        rec = payload.get("recommendation", {})
        if not ticker or not isinstance(rec, dict):
            continue
        total += 1
        generated_at = str(payload.get("generated_at", ""))
        # Fallback to file mtime if generated_at missing.
        if not generated_at:
            generated_at = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
        current = latest.get(ticker)
        if current is None or generated_at >= current[0]:
            latest[ticker] = (generated_at, payload)
    return {k: v[1] for k, v in latest.items()}, total


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


def _settlement_horizon_days(record: dict) -> int:
    value = _safe_float(record.get("horizon_days"), 0.0)
    if value > 0:
        return int(value)
    horizon = str(record.get("horizon", "")).upper().replace(" ", "")
    if horizon.startswith("T+"):
        try:
            return int(horizon.split("T+", 1)[1])
        except Exception:
            pass
    settlement_key = str(record.get("settlement_key", "")).upper()
    if "::T+" in settlement_key:
        try:
            return int(settlement_key.rsplit("::T+", 1)[1])
        except Exception:
            pass
    return 1


def _compact_settlement_record(record: dict) -> dict:
    horizon_days = _settlement_horizon_days(record)
    return {
        "settled": True,
        "horizon": str(record.get("horizon") or f"T+{horizon_days}"),
        "horizon_days": horizon_days,
        "pnl_percent": round(_safe_float(record.get("pnl_percent"), 0.0), 4),
        "market_move_percent": round(_safe_float(record.get("market_move_percent"), 0.0), 4),
        "excess_return_vs_best_baseline": (
            round(_safe_float(record.get("excess_return_vs_best_baseline"), 0.0), 4)
            if record.get("excess_return_vs_best_baseline") is not None
            else None
        ),
        "decision": str(record.get("decision", "")).upper(),
        "as_of_date": record.get("as_of_date", ""),
        "settled_date": record.get("settled_date", ""),
        "settled_at": record.get("settled_at", ""),
        "advice_file": record.get("advice_file", ""),
    }


def _latest_settlements_by_ticker() -> dict[str, dict]:
    state = read_json(ADVICE_SETTLEMENT_PATH, {})
    records = state.get("records", []) if isinstance(state, dict) else []
    if not isinstance(records, list):
        return {}

    latest: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict) or str(record.get("source", "")).strip() != "advice_settlement":
            continue
        ticker = _normalize_ticker(record.get("ticker", ""))
        if not ticker:
            continue
        compact = _compact_settlement_record(record)
        current = latest.get(ticker)
        sort_key = (
            str(compact.get("as_of_date") or ""),
            int(compact.get("horizon_days") or 0),
            str(compact.get("settled_date") or ""),
            str(compact.get("settled_at") or ""),
        )
        if current is None or sort_key >= current["_sort_key"]:
            compact["_sort_key"] = sort_key
            latest[ticker] = compact

    for row in latest.values():
        row.pop("_sort_key", None)
    return latest


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
        ticker = _normalize_ticker(row.get("ticker", "")) or "UNKNOWN"
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


def _advice_settlement_stats(total_files: int | None = None) -> dict:
    state = read_json(ADVICE_SETTLEMENT_PATH, {})
    if not isinstance(state, dict):
        state = {}
    settled = state.get("settled_keys", [])
    settled_set = set(str(x) for x in settled) if isinstance(settled, list) else set()
    if total_files is None:
        _, total_files = _collect_advice_snapshot()
    settled_count = sum(1 for x in settled_set if "::T+" in x) + sum(1 for x in settled_set if "::T+" not in x)
    expected_count = max(0, int(total_files or 0)) * len(_ADVICE_EVALUATION_HORIZONS)
    pending_count = max(0, expected_count - settled_count)
    return {
        "settled_count": settled_count,
        "pending_count": pending_count,
        "horizons": [f"T+{x}" for x in _ADVICE_EVALUATION_HORIZONS],
        "evaluation": state.get("evaluation", {}) if isinstance(state.get("evaluation"), dict) else {},
        "updated_at": state.get("updated_at", ""),
    }


def build_dashboard_summary() -> dict:
    latest_advice, advice_total_count = _collect_advice_snapshot()
    latest_settlements = _latest_settlements_by_ticker()
    advice_actions = Counter()
    advice_rows: list[dict] = []
    for ticker, payload in latest_advice.items():
        rec = payload.get("recommendation", {})
        multi_period = payload.get("multi_period_advice", {}) or {}
        short_term = multi_period.get("short_term") or payload.get("short_term") or {}
        swing_term = multi_period.get("swing_term") or payload.get("swing_term") or {}
        action = str(rec.get("action", "HOLD")).upper()
        advice_actions[action] += 1
        row = {
            "ticker": ticker,
            "generated_at": payload.get("generated_at"),
            "action": action,
            "position_percent": _safe_float(rec.get("position_percent"), 0.0),
            "confidence": _safe_float(rec.get("confidence"), 0.0),
            "reason": rec.get("reason", ""),
            "short_term_action": str(short_term.get("action", "")).upper() or "",
            "swing_term_action": str(swing_term.get("action", "")).upper() or "",
        }
        settlement = latest_settlements.get(ticker)
        if settlement:
            row["settlement"] = settlement
            row["settled"] = True
            row["settled_horizon"] = settlement.get("horizon", "")
            row["settled_return_pct"] = settlement.get("pnl_percent")
            row["settled_market_move_pct"] = settlement.get("market_move_percent")
            row["settled_excess_best_pct"] = settlement.get("excess_return_vs_best_baseline")
            row["settled_date"] = settlement.get("settled_date", "")
            row["settlement_as_of_date"] = settlement.get("as_of_date", "")
        else:
            row["settled"] = False
        advice_rows.append(row)
    advice_rows.sort(key=lambda x: x["generated_at"] or "", reverse=True)

    summary = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "advice_total_count": advice_total_count,
        "latest_advice_count": len(latest_advice),
        "advice_action_distribution": dict(advice_actions),
        "latest_advice": advice_rows[:50],
        "recent_backtest_runs": _collect_latest_backtest_runs(),
        "reflection_stats": _reflection_stats(),
        "advice_settlement_stats": _advice_settlement_stats(advice_total_count),
    }
    write_json(INDEX_SUMMARY_PATH, summary)
    return summary
