from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime
from typing import Any

from .config import STRATEGY_RULES_PATH
from .storage import read_json, write_json


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _clip_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _normalize_action(value: Any) -> str:
    token = str(value or "").upper()
    if "BUY" in token:
        return "BUY"
    if "SELL" in token:
        return "SELL"
    return "HOLD"


def _normalize_rule_key_text(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())[:64]


def _rule_id(*parts: Any) -> str:
    raw = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"sr_{digest}"


def _display_group_key(rule: dict) -> str:
    return "|".join(
        [
            str(rule.get("mistake_type", "") or "").strip(),
            _normalize_action(rule.get("action", "HOLD")),
            _normalize_rule_key_text(rule.get("rule_text", "")),
        ]
    )


def _severity_weight(severity: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(str(severity or "").lower(), 1)


def _stock_category_from_signal(record: dict) -> str:
    profile = record.get("signal_profile") if isinstance(record.get("signal_profile"), dict) else {}
    category = str(profile.get("stock_category", "") or "").strip()
    return category or "all"


def load_strategy_rules() -> dict:
    payload = read_json(STRATEGY_RULES_PATH, {})
    if not isinstance(payload, dict):
        payload = {}
    rules = payload.get("rules")
    if not isinstance(rules, list):
        rules = []
    return {
        "version": int(payload.get("version", 1) or 1),
        "updated_at": str(payload.get("updated_at", "") or ""),
        "rules": [rule for rule in rules if isinstance(rule, dict)],
        "summary": payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {},
    }


def _save_strategy_rules(payload: dict) -> None:
    payload = dict(payload)
    payload["version"] = int(payload.get("version", 1) or 1)
    payload["updated_at"] = _now()
    write_json(STRATEGY_RULES_PATH, payload)


def _rule_effect_for_mistake(mistake_type: str) -> dict:
    mistake_type = str(mistake_type or "").strip()
    if mistake_type == "low_quality_overreach":
        return {
            "confidence_cap": 0.55,
            "position_cap": 20.0,
            "position_multiplier": 0.6,
            "action_policy": "cap_or_hold_low_quality",
        }
    if mistake_type == "false_breakout":
        return {
            "confidence_delta": -0.05,
            "position_multiplier": 0.75,
            "action_policy": "require_breakout_confirmation",
        }
    if mistake_type == "over_defensive_sell":
        return {
            "confidence_delta": -0.04,
            "position_multiplier": 0.5,
            "action_policy": "avoid_full_sell_without_breakdown",
        }
    if mistake_type == "directional_miss":
        return {
            "confidence_delta": -0.05,
            "position_multiplier": 0.8,
            "action_policy": "discount_conflicting_direction",
        }
    if mistake_type == "missed_opportunity":
        return {
            "confidence_delta": -0.03,
            "position_cap": 8.0,
            "action_policy": "allow_probe_when_quality_and_trend_align",
        }
    if mistake_type == "avoided_loss":
        return {
            "confidence_delta": 0.02,
            "action_policy": "preserve_defensive_hold",
        }
    return {"action_policy": "observe"}


def _build_rule_from_record(record: dict) -> dict | None:
    attribution = record.get("mistake_attribution") if isinstance(record.get("mistake_attribution"), dict) else {}
    mistake_type = str(attribution.get("mistake_type", "") or "").strip()
    if not mistake_type or mistake_type == "no_mistake":
        return None

    future_rule = _clip_text(attribution.get("future_rule"), 220)
    if not future_rule:
        return None

    action = _normalize_action(record.get("decision", "HOLD"))
    horizon = str(record.get("horizon", "") or "").strip() or "all"
    stock_category = _stock_category_from_signal(record)
    key_text = _normalize_rule_key_text(future_rule)
    rule_id = _rule_id(mistake_type, action, horizon, stock_category, key_text)
    severity = str(attribution.get("severity", "") or "low").lower()
    return {
        "id": rule_id,
        "status": "active",
        "rule_text": future_rule,
        "mistake_type": mistake_type,
        "action": action,
        "horizon": horizon,
        "stock_category": stock_category,
        "tags": list(dict.fromkeys([str(x) for x in attribution.get("tags", []) if str(x).strip()])),
        "effect": _rule_effect_for_mistake(mistake_type),
        "support_count": 1,
        "severity_score": _severity_weight(severity),
        "severity_distribution": {severity: 1},
        "avg_pnl_percent": round(_safe_float(record.get("pnl_percent"), 0.0), 4),
        "avg_excess_vs_best_baseline": round(_safe_float(record.get("excess_return_vs_best_baseline"), 0.0), 4),
        "latest_root_cause": _clip_text(attribution.get("root_cause"), 220),
        "last_seen_at": str(record.get("settled_at", "") or _now()),
        "created_at": _now(),
        "updated_at": _now(),
        "source_examples": [
            {
                "ticker": str(record.get("ticker", "") or "").zfill(6),
                "horizon": horizon,
                "decision": action,
                "pnl_percent": round(_safe_float(record.get("pnl_percent"), 0.0), 4),
                "settled_date": str(record.get("settled_date", "") or ""),
                "root_cause": _clip_text(attribution.get("root_cause"), 160),
            }
        ],
    }


def _merge_rule(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    old_count = max(0, int(merged.get("support_count", 0) or 0))
    new_count = max(1, int(incoming.get("support_count", 1) or 1))
    total = old_count + new_count
    for metric in ("avg_pnl_percent", "avg_excess_vs_best_baseline"):
        old_value = _safe_float(merged.get(metric), 0.0)
        new_value = _safe_float(incoming.get(metric), 0.0)
        merged[metric] = round(((old_value * old_count) + (new_value * new_count)) / total, 4)

    merged["support_count"] = total
    merged["severity_score"] = int(merged.get("severity_score", 0) or 0) + int(incoming.get("severity_score", 0) or 0)
    severity_distribution = defaultdict(int)
    for source in (merged.get("severity_distribution", {}), incoming.get("severity_distribution", {})):
        if isinstance(source, dict):
            for key, value in source.items():
                severity_distribution[str(key)] += int(value or 0)
    merged["severity_distribution"] = dict(sorted(severity_distribution.items()))
    merged["tags"] = list(dict.fromkeys((merged.get("tags") or []) + (incoming.get("tags") or [])))[:10]
    merged["latest_root_cause"] = incoming.get("latest_root_cause") or merged.get("latest_root_cause", "")
    merged["last_seen_at"] = incoming.get("last_seen_at") or _now()
    merged["updated_at"] = _now()
    examples = (merged.get("source_examples") or []) + (incoming.get("source_examples") or [])
    merged["source_examples"] = examples[-8:]
    merged["status"] = "active" if merged["severity_score"] >= 2 or merged["support_count"] >= 2 else "watch"
    return merged


def _merge_rule_for_display(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    old_count = max(0, int(merged.get("support_count", 0) or 0))
    new_count = max(1, int(incoming.get("support_count", 1) or 1))
    total = old_count + new_count
    for metric in ("avg_pnl_percent", "avg_excess_vs_best_baseline"):
        old_value = _safe_float(merged.get(metric), 0.0)
        new_value = _safe_float(incoming.get(metric), 0.0)
        merged[metric] = round(((old_value * old_count) + (new_value * new_count)) / total, 4)

    merged["support_count"] = total
    merged["severity_score"] = int(merged.get("severity_score", 0) or 0) + int(incoming.get("severity_score", 0) or 0)
    merged["status"] = "active" if "active" in {merged.get("status"), incoming.get("status")} else merged.get("status", "watch")
    merged["source_rule_count"] = int(merged.get("source_rule_count", 1) or 1) + 1
    merged["source_rule_ids"] = list(
        dict.fromkeys((merged.get("source_rule_ids") or [merged.get("id")]) + [incoming.get("id")])
    )
    merged["horizons"] = list(
        dict.fromkeys((merged.get("horizons") or [merged.get("horizon")]) + [incoming.get("horizon")])
    )
    categories = list(dict.fromkeys((merged.get("stock_categories") or [merged.get("stock_category")]) + [incoming.get("stock_category")]))
    categories = [str(x) for x in categories if str(x or "").strip()]
    merged["stock_categories"] = categories
    merged["stock_category"] = " / ".join(categories[:4]) + (" ..." if len(categories) > 4 else "") if categories else "all"
    examples = (merged.get("source_examples") or []) + (incoming.get("source_examples") or [])
    merged["source_examples"] = examples[-12:]
    merged["last_seen_at"] = max(str(merged.get("last_seen_at", "")), str(incoming.get("last_seen_at", "")))
    merged["updated_at"] = max(str(merged.get("updated_at", "")), str(incoming.get("updated_at", "")))
    return merged


def group_strategy_rules_for_display(rules: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        key = _display_group_key(rule)
        item = dict(rule)
        item.setdefault("source_rule_count", 1)
        item.setdefault("source_rule_ids", [item.get("id")])
        item.setdefault("horizons", [item.get("horizon")])
        item.setdefault("stock_categories", [item.get("stock_category")])
        if key in grouped:
            grouped[key] = _merge_rule_for_display(grouped[key], item)
        else:
            grouped[key] = item
    return sorted(
        grouped.values(),
        key=lambda rule: (int(rule.get("severity_score", 0) or 0), int(rule.get("support_count", 0) or 0), str(rule.get("last_seen_at", ""))),
        reverse=True,
    )


def update_strategy_rules_from_records(records: list[dict]) -> dict:
    """Persist settlement mistake attributions as reusable strategy rules."""
    payload = load_strategy_rules()
    rules_by_id = {str(rule.get("id", "")): rule for rule in payload.get("rules", []) if str(rule.get("id", "")).strip()}
    inserted = 0
    updated = 0
    for record in records or []:
        if not isinstance(record, dict):
            continue
        incoming = _build_rule_from_record(record)
        if not incoming:
            continue
        rule_id = incoming["id"]
        if rule_id in rules_by_id:
            rules_by_id[rule_id] = _merge_rule(rules_by_id[rule_id], incoming)
            updated += 1
        else:
            rules_by_id[rule_id] = incoming
            inserted += 1

    rules = sorted(
        rules_by_id.values(),
        key=lambda rule: (int(rule.get("severity_score", 0) or 0), int(rule.get("support_count", 0) or 0), str(rule.get("last_seen_at", ""))),
        reverse=True,
    )
    summary = {
        "total_rules": len(rules),
        "active_rules": sum(1 for rule in rules if rule.get("status") == "active"),
        "watch_rules": sum(1 for rule in rules if rule.get("status") == "watch"),
        "by_mistake_type": {},
    }
    by_type = defaultdict(int)
    for rule in rules:
        by_type[str(rule.get("mistake_type", "unknown"))] += 1
    summary["by_mistake_type"] = dict(sorted(by_type.items(), key=lambda item: item[1], reverse=True))
    next_payload = {
        "version": payload.get("version", 1),
        "updated_at": _now(),
        "rules": rules[:500],
        "summary": summary,
    }
    if inserted or updated:
        _save_strategy_rules(next_payload)
    return {
        "inserted": inserted,
        "updated": updated,
        "total_rules": len(next_payload["rules"]),
        "active_rules": summary["active_rules"],
        "path": STRATEGY_RULES_PATH,
    }


def _advice_stock_category(ticker: str) -> str:
    ticker = str(ticker or "").zfill(6)
    if ticker.startswith(("688", "689")):
        return "科创板"
    if ticker.startswith(("300", "301")):
        return "创业板"
    if ticker.startswith(("8", "4")):
        return "北交所"
    if ticker.startswith(("600", "601", "603", "605")):
        return "沪市主板"
    if ticker.startswith(("000", "001", "002", "003")):
        return "深市主板"
    return "其他"


def _signal_action(case: dict) -> str:
    value = str((case or {}).get("sentiment") or (case or {}).get("decision") or (case or {}).get("action") or "").lower()
    if any(token in value for token in ("positive", "bull", "buy", "看多", "买入", "增持")):
        return "BUY"
    if any(token in value for token in ("negative", "bear", "sell", "看空", "卖出", "减持")):
        return "SELL"
    return _normalize_action(value)


def _score_rule_applicability(rule: dict, advice: dict) -> float:
    rec = advice.get("recommendation", {}) if isinstance(advice.get("recommendation"), dict) else {}
    action = _normalize_action(rec.get("action", "HOLD"))
    ticker = str(advice.get("ticker", "") or "")
    stock_category = _advice_stock_category(ticker)
    rule_action = _normalize_action(rule.get("action", "HOLD"))
    rule_category = str(rule.get("stock_category", "all") or "all")
    mistake_type = str(rule.get("mistake_type", "") or "")
    score = 0.0
    if rule.get("status") != "active":
        score -= 0.2
    if rule_action == action:
        score += 0.5
    elif mistake_type == "missed_opportunity" and action == "HOLD":
        score += 0.35
    else:
        score -= 0.35
    if rule_category in {"", "all", stock_category}:
        score += 0.25
    else:
        score -= 0.15
    score += min(0.25, int(rule.get("support_count", 0) or 0) * 0.05)
    score += min(0.25, int(rule.get("severity_score", 0) or 0) * 0.03)
    return round(score, 4)


def get_applicable_strategy_rules(
    advice: dict,
    limit: int = 5,
    allowed_rule_ids: set[str] | None = None,
    disabled_rule_ids: set[str] | None = None,
) -> list[dict]:
    payload = load_strategy_rules()
    candidates = []
    disabled_rule_ids = disabled_rule_ids or set()
    for rule in payload.get("rules", []):
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id", ""))
        if rule_id in disabled_rule_ids:
            continue
        if allowed_rule_ids is not None and rule_id not in allowed_rule_ids:
            continue
        score = _score_rule_applicability(rule, advice)
        if score <= 0.15:
            continue
        item = dict(rule)
        item["applicability_score"] = score
        candidates.append(item)
    candidates.sort(key=lambda rule: (rule.get("applicability_score", 0), int(rule.get("severity_score", 0) or 0)), reverse=True)
    return candidates[: max(1, int(limit))]


def apply_strategy_rules_to_advice(
    advice: dict,
    *,
    limit: int = 5,
    allowed_rule_ids: set[str] | None = None,
    disabled_rule_ids: set[str] | None = None,
) -> dict:
    """Apply learned rules as a conservative post-processor for future advice."""
    if not isinstance(advice, dict):
        return advice
    recommendation = advice.get("recommendation") if isinstance(advice.get("recommendation"), dict) else {}
    if not recommendation:
        return advice
    existing_rule_state = advice.get("strategy_rules") if isinstance(advice.get("strategy_rules"), dict) else {}
    if existing_rule_state.get("applied"):
        return advice
    risk = advice.get("risk") if isinstance(advice.get("risk"), dict) else {}
    analysts = advice.get("analyst_cases") if isinstance(advice.get("analyst_cases"), dict) else {}
    data_quality = advice.get("data_quality") if isinstance(advice.get("data_quality"), dict) else {}
    referee = advice.get("referee") if isinstance(advice.get("referee"), dict) else {}
    technical = analysts.get("technical_flow") if isinstance(analysts.get("technical_flow"), dict) else {}
    fundamental = analysts.get("fundamental_news") if isinstance(analysts.get("fundamental_news"), dict) else {}

    action = _normalize_action(recommendation.get("action", "HOLD"))
    original_action = action
    confidence = max(0.0, min(1.0, _safe_float(recommendation.get("confidence"), 0.5)))
    position = max(0.0, min(100.0, _safe_float(recommendation.get("position_percent"), risk.get("position_percent", 0.0))))
    quality_score = _safe_float(data_quality.get("score"), 0.5)
    trend_strength = _safe_float(referee.get("trend_strength"), 0.0)
    tech_action = _signal_action(technical)
    fund_action = _signal_action(fundamental)
    conflict = tech_action in {"BUY", "SELL"} and fund_action in {"BUY", "SELL"} and tech_action != fund_action

    applied: list[dict] = []
    notes: list[str] = []
    for rule in get_applicable_strategy_rules(
        advice,
        limit=limit,
        allowed_rule_ids=allowed_rule_ids,
        disabled_rule_ids=disabled_rule_ids,
    ):
        mistake_type = str(rule.get("mistake_type", "") or "")
        effect = rule.get("effect") if isinstance(rule.get("effect"), dict) else {}
        used = False
        if mistake_type == "low_quality_overreach" and action in {"BUY", "SELL"} and quality_score < 0.45:
            confidence = min(confidence, _safe_float(effect.get("confidence_cap"), 0.55))
            position = min(position, _safe_float(effect.get("position_cap"), 20.0))
            if confidence < 0.48:
                action = "HOLD"
                position = 0.0
            used = True
        elif mistake_type == "false_breakout" and action == "BUY":
            confidence = max(0.2, confidence + _safe_float(effect.get("confidence_delta"), -0.05))
            position *= _safe_float(effect.get("position_multiplier"), 0.75)
            used = True
        elif mistake_type == "over_defensive_sell" and action == "SELL":
            features = advice.get("technical_features") if isinstance(advice.get("technical_features"), dict) else {}
            trend = features.get("trend") if isinstance(features.get("trend"), dict) else {}
            price_vs_ma20 = _safe_float(trend.get("price_vs_ma20_pct"), -99.0)
            ma20_slope = _safe_float(trend.get("ma20_slope_5d_pct"), -99.0)
            if price_vs_ma20 > -2.0 and ma20_slope > -1.0:
                action = "HOLD"
                position *= _safe_float(effect.get("position_multiplier"), 0.5)
                confidence = max(0.2, confidence + _safe_float(effect.get("confidence_delta"), -0.04))
                used = True
        elif mistake_type == "directional_miss" and action in {"BUY", "SELL"} and conflict:
            confidence = max(0.2, confidence + _safe_float(effect.get("confidence_delta"), -0.05))
            position *= _safe_float(effect.get("position_multiplier"), 0.8)
            used = True
        elif mistake_type == "missed_opportunity" and action == "HOLD":
            if quality_score >= 0.60 and trend_strength >= 0.60 and tech_action == "BUY" and fund_action in {"BUY", "HOLD"}:
                action = "BUY"
                confidence = max(confidence, 0.52)
                position = max(position, min(8.0, _safe_float(effect.get("position_cap"), 8.0)))
                used = True
        elif mistake_type == "avoided_loss" and action == "HOLD":
            confidence = min(0.95, confidence + _safe_float(effect.get("confidence_delta"), 0.02))
            used = True

        if used:
            applied.append(
                {
                    "id": rule.get("id"),
                    "mistake_type": mistake_type,
                    "rule_text": rule.get("rule_text"),
                    "support_count": rule.get("support_count", 0),
                    "applicability_score": rule.get("applicability_score"),
                }
            )
            notes.append(_clip_text(rule.get("rule_text"), 120))

    if not applied:
        advice["strategy_rules"] = {
            "applied": False,
            "candidate_count": len(
                get_applicable_strategy_rules(
                    advice,
                    limit=limit,
                    allowed_rule_ids=allowed_rule_ids,
                    disabled_rule_ids=disabled_rule_ids,
                )
            ),
            "applied_rules": [],
        }
        return advice

    if action in {"BUY", "SELL"} and position <= 0:
        action = "HOLD"
        notes.append("规则校准后方向性仓位为0，执行动作转为HOLD。")

    recommendation["action"] = action
    recommendation["confidence"] = round(max(0.0, min(1.0, confidence)), 4)
    recommendation["position_percent"] = round(max(0.0, min(100.0, position)), 2)
    recommendation["execution_action"] = action if action == "HOLD" else f"{action} {recommendation['position_percent']:.1f}%"
    reason = str(recommendation.get("reason", "") or "").strip()
    rule_note = f"策略规则库校准：{'；'.join(notes[:3])}"
    recommendation["reason"] = f"{reason}｜{rule_note}" if reason else rule_note
    recommendation["strategy_rule_adjusted"] = True

    if risk:
        risk["position_percent"] = recommendation["position_percent"]
        risk["final_action"] = action
        risk["action"] = action
        risk["decision"] = action
        risk_reason = str(risk.get("reason", "") or "").strip()
        risk["reason"] = f"{risk_reason}；{rule_note}" if risk_reason else rule_note
        advice["risk"] = risk
    advice["strategy_rules"] = {
        "applied": True,
        "original_action": original_action,
        "adjusted_action": action,
        "applied_rules": applied,
    }
    return advice
