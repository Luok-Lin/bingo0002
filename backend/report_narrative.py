from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any

from agents.llm_client import LLMClient, looks_unconfigured_secret, repair_json_text


NARRATIVE_KEYS = (
    "executive_summary",
    "investment_thesis",
    "evidence_review",
    "risk_review",
    "stability_review",
    "execution_plan",
    "watchlist",
)

FORBIDDEN_FIELD_TOKENS = (
    "risk_profile",
    "holding_period",
    "max_position_per_stock",
    "already_holding",
    "current_position_percent",
    "cost_price",
    "prefer_stop_loss",
    "personalization",
    "applied",
    "original_action",
    "adjusted_action",
    "parse_fail_count",
    "rule_fallback_count",
    "empty_reason_count",
    "consensus_tech_fund",
    "referee_action",
    "risk_action",
    "recommendation_action",
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int, lower: int = 1, upper: int = 4096) -> int:
    try:
        value = int(str(os.getenv(name, str(default))).strip() or str(default))
    except ValueError:
        value = default
    return max(lower, min(upper, value))


def _clean(value: Any, default: str = "") -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text or default


def _clip(value: Any, limit: int = 500) -> str:
    text = _clean(value)
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
        if number == number and number not in {float("inf"), float("-inf")}:
            return number
    except Exception:
        pass
    return default


def _fmt_number(value: Any, digits: int = 2, default: str = "-") -> str:
    number = _safe_float(value)
    return f"{number:.{digits}f}" if number is not None else default


def _fmt_percent(value: Any, digits: int = 2, default: str = "-") -> str:
    number = _safe_float(value)
    return f"{number:.{digits}f}%" if number is not None else default


def _join_items(items: Any, limit: int = 5) -> str:
    if not isinstance(items, list):
        return _clean(items, "-")
    values = [_clean(item) for item in items if _clean(item)]
    return "；".join(values[:limit]) or "-"


def _normalize_action(value: Any) -> str:
    token = _clean(value, "HOLD").upper()
    if "BUY" in token:
        return "BUY"
    if "SELL" in token:
        return "SELL"
    return "HOLD"


def _source_type_text(source_types: dict) -> str:
    labels = {
        "news": "个股新闻",
        "market_news": "市场要闻",
        "report": "券商研报",
        "announcement": "公司公告",
        "financial_abstract": "财务摘要",
        "financial_indicator": "财务指标",
        "income_statement": "利润表",
        "balance_sheet": "资产负债表",
        "cash_flow": "现金流量表",
        "fallback": "兜底文本",
    }
    rows = []
    for key, value in sorted((source_types or {}).items(), key=lambda item: (-int(item[1] or 0), str(item[0]))):
        try:
            count = int(value or 0)
        except Exception:
            count = 0
        if count > 0:
            rows.append(f"{labels.get(str(key), str(key))}{count}条")
    return "，".join(rows[:8]) or "暂无来源统计"


def _stability_text(advice: dict) -> str:
    stability = advice.get("stability_diagnostics", {}) if isinstance(advice.get("stability_diagnostics"), dict) else {}
    score = _fmt_number(advice.get("stability_score"))
    level = _clean(advice.get("stability_level"), "未评级")
    parse_fail = int(_safe_float(stability.get("parse_fail_count"), 0.0) or 0)
    fallback = int(_safe_float(stability.get("rule_fallback_count"), 0.0) or 0)
    empty = int(_safe_float(stability.get("empty_reason_count"), 0.0) or 0)
    consensus = stability.get("consensus_tech_fund")
    parts = [f"稳定度 {score}，等级 {level}。"]
    if parse_fail == 0 and empty == 0:
        parts.append("分析链路解析完整。")
    else:
        parts.append("部分分析输出完整度不足，最终建议已降低相关权重。")
    if fallback:
        parts.append("存在规则兜底或保护性降级。")
    if consensus is True:
        parts.append("技术面与基本面方向一致。")
    elif consensus is False:
        parts.append("技术面与基本面存在分歧。")
    note = _clean(advice.get("stability_note"))
    if note:
        parts.append(note)
    return "".join(parts)


def build_report_fact_sheet(advice: dict) -> dict:
    rec = advice.get("recommendation", {}) if isinstance(advice.get("recommendation"), dict) else {}
    risk = advice.get("risk", {}) if isinstance(advice.get("risk"), dict) else {}
    referee = advice.get("referee", {}) if isinstance(advice.get("referee"), dict) else {}
    data_quality = advice.get("data_quality", {}) if isinstance(advice.get("data_quality"), dict) else {}
    latest = advice.get("latest_market", {}) if isinstance(advice.get("latest_market"), dict) else {}
    multi = advice.get("multi_period_advice", {}) if isinstance(advice.get("multi_period_advice"), dict) else {}
    short_term = multi.get("short_term") or advice.get("short_term") or {}
    swing_term = multi.get("swing_term") or advice.get("swing_term") or {}
    risk_plan = multi.get("risk_plan") or advice.get("risk_plan") or {}
    features = advice.get("technical_features", {}) if isinstance(advice.get("technical_features"), dict) else {}
    rag = data_quality.get("rag", {}) if isinstance(data_quality.get("rag"), dict) else {}
    market = data_quality.get("market", {}) if isinstance(data_quality.get("market"), dict) else {}
    analysts = advice.get("analyst_cases", {}) if isinstance(advice.get("analyst_cases"), dict) else {}

    analyst_rows = []
    for name, case in analysts.items():
        if not isinstance(case, dict):
            continue
        analyst_rows.append(
            {
                "role": _clean(case.get("agent") or name),
                "view": _clean(case.get("sentiment") or case.get("decision"), "neutral"),
                "confidence": _fmt_number(case.get("confidence")),
                "reason": _clip(case.get("reasoning") or case.get("thought_process"), 260),
            }
        )

    return {
        "ticker": _clean(advice.get("ticker"), "UNKNOWN"),
        "as_of_date": _clean(advice.get("as_of_date")),
        "generated_at": _clean(advice.get("generated_at")),
        "final_action": _normalize_action(rec.get("action")),
        "execution_action": _normalize_action(rec.get("execution_action") or risk.get("final_action") or rec.get("action")),
        "position_percent": _fmt_percent(rec.get("position_percent")),
        "confidence": _fmt_number(rec.get("confidence")),
        "recommendation_reason": _clip(rec.get("reason") or rec.get("thesis") or rec.get("rationale"), 360),
        "risk_decision": _normalize_action(risk.get("decision") or risk.get("action") or risk.get("final_action")),
        "risk_reason": _clip(risk.get("reason") or risk.get("decision") or risk.get("action"), 360),
        "referee_decision": _normalize_action(referee.get("decision")),
        "referee_reason": _clip(referee.get("reason") or referee.get("sentiment"), 320),
        "market_snapshot": {
            "source": _clean(market.get("source") or latest.get("source"), "-"),
            "as_of_date": _clean(market.get("as_of_date") or latest.get("as_of_date"), "-"),
            "close": _fmt_number(latest.get("close")),
            "pct_change": _fmt_percent(latest.get("pct_change")),
            "turnover": _fmt_percent(latest.get("turnover")),
            "rows": _clean(market.get("rows"), "-"),
        },
        "data_quality": {
            "score": _fmt_number(data_quality.get("score")),
            "level": _clean(data_quality.get("level"), "-"),
            "note": _clip(data_quality.get("note"), 260),
        },
        "evidence": {
            "documents": _clean(rag.get("documents"), "0"),
            "latest_date": _clean(rag.get("latest_date"), "-"),
            "status": _clean(rag.get("status"), "-"),
            "source_mix": _source_type_text(rag.get("source_types", {}) if isinstance(rag.get("source_types"), dict) else {}),
        },
        "technical": {
            "signals": _join_items(features.get("signals", []), 6),
            "trend": _clip(features.get("summary_text") or features.get("trend"), 280),
        },
        "analyst_views": analyst_rows[:6],
        "period_plan": {
            "short_term": {
                "action": _normalize_action(short_term.get("action") or rec.get("action")),
                "confidence": _fmt_number(short_term.get("confidence")),
                "reason": _clip(short_term.get("reason"), 260),
            },
            "swing_term": {
                "action": _normalize_action(swing_term.get("action") or rec.get("action")),
                "confidence": _fmt_number(swing_term.get("confidence")),
                "reason": _clip(swing_term.get("reason"), 260),
            },
            "stop_loss": _clip(risk_plan.get("stop_loss"), 180),
            "take_profit": _clip(risk_plan.get("take_profit"), 180),
            "invalid_conditions": _join_items(risk_plan.get("invalid_conditions"), 5),
        },
        "stability": _stability_text(advice),
        "warnings": [_clip(item, 180) for item in advice.get("warnings", [])[:5] if _clean(item)],
    }


def deterministic_report_narrative(advice: dict) -> dict:
    facts = build_report_fact_sheet(advice)
    action = facts["final_action"]
    execution_action = facts["execution_action"]
    position = facts["position_percent"]
    quality = facts["data_quality"]
    evidence = facts["evidence"]
    period = facts["period_plan"]
    risk = facts["risk_reason"] or "暂无额外风控说明。"

    if action == execution_action:
        summary = f"{facts['ticker']} 当前建议为 {action}，执行层保持一致，建议仓位约 {position}，置信度 {facts['confidence']}。"
    else:
        summary = f"{facts['ticker']} 策略层为 {action}，执行层按风控约束调整为 {execution_action}，建议仓位约 {position}。"

    return {
        "executive_summary": (
            f"{summary} 数据质量为 {quality['score']} / {quality['level']}，"
            f"证据覆盖 {evidence['documents']} 份文档，来源结构为：{evidence['source_mix']}。"
        ),
        "investment_thesis": facts["recommendation_reason"] or facts["referee_reason"] or "当前多空信号尚未形成足够优势，建议以执行纪律为主。",
        "evidence_review": (
            f"行情来源为 {facts['market_snapshot']['source']}，最近日期 {facts['market_snapshot']['as_of_date']}；"
            f"RAG 证据状态为 {evidence['status']}，最新证据日期 {evidence['latest_date']}。"
            f"分析师观点：{'; '.join(item['role'] + ' ' + item['view'] + ' ' + item['confidence'] for item in facts['analyst_views']) or '暂无完整分析师摘要'}。"
        ),
        "risk_review": f"{risk} 止损参考：{period['stop_loss'] or '-'}；止盈参考：{period['take_profit'] or '-'}。",
        "stability_review": facts["stability"],
        "execution_plan": (
            f"短线计划为 {period['short_term']['action']}，中线计划为 {period['swing_term']['action']}。"
            f"若触发失效条件：{period['invalid_conditions']}，应重新评估仓位与方向。"
        ),
        "watchlist": [
            "盘中确认量能是否延续，避免在波动放大时追价。",
            "跟踪公告、研报与新闻是否改变核心假设。",
            "复盘 T+1/T+3/T+5 表现，用于后续策略校准。",
        ],
    }


def _has_forbidden_tokens(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False).lower()
    return any(token.lower() in text for token in FORBIDDEN_FIELD_TOKENS)


def _normalize_narrative(payload: dict, fallback: dict) -> dict:
    normalized = {}
    for key in NARRATIVE_KEYS:
        value = payload.get(key)
        if key == "watchlist":
            if not isinstance(value, list):
                value = fallback.get(key, [])
            items = [_clip(item, 120) for item in value if _clean(item)]
            normalized[key] = items[:5] or fallback.get(key, [])
        else:
            text = _clip(value, 900)
            normalized[key] = text or fallback.get(key, "")
    if _has_forbidden_tokens(normalized):
        return fallback
    return normalized


def _build_prompt(facts: dict) -> str:
    return (
        "请基于下方白名单事实，生成一份中文投研报告叙述层。"
        "只能解释事实，不得创造新的价格、仓位、评级、来源或止损止盈条件；"
        "不得输出任何字段名、内部诊断键名、英文键名或 JSON 以外的文字。"
        "如果策略层与执行层不一致，必须说明以风控执行动作为准。\n\n"
        f"白名单事实：{json.dumps(facts, ensure_ascii=False)}\n\n"
        "输出纯 JSON，结构必须为："
        "{\"executive_summary\":\"两到四句执行摘要\","
        "\"investment_thesis\":\"核心逻辑\","
        "\"evidence_review\":\"证据链与数据来源说明\","
        "\"risk_review\":\"风险与风控说明\","
        "\"stability_review\":\"稳定度解释\","
        "\"execution_plan\":\"执行计划\","
        "\"watchlist\":[\"观察点1\",\"观察点2\",\"观察点3\"]}"
    )


def build_report_narrative(advice: dict, *, use_llm: bool | None = None) -> dict:
    fallback = deterministic_report_narrative(advice)
    enabled = _env_bool("ENABLE_LLM_REPORT_NARRATIVE", True) if use_llm is None else bool(use_llm)
    if not enabled:
        return {**fallback, "_source": "deterministic"}

    base_config = LLMClient.from_env()
    if looks_unconfigured_secret(base_config.api_key):
        return {**fallback, "_source": "deterministic_unconfigured"}

    client = LLMClient(
        replace(
            base_config,
            timeout_seconds=_env_int("REPORT_NARRATIVE_LLM_TIMEOUT_SECONDS", 25, lower=5, upper=120),
            json_retry=_env_int("REPORT_NARRATIVE_LLM_JSON_RETRY", 1, lower=1, upper=3),
            max_tokens=_env_int("REPORT_NARRATIVE_LLM_MAX_TOKENS", 1200, lower=300, upper=3000),
            response_format_json=_env_bool("REPORT_NARRATIVE_LLM_RESPONSE_FORMAT_JSON", True),
        )
    )
    facts = build_report_fact_sheet(advice)
    try:
        raw = client.query(_build_prompt(facts), role="投研报告撰写助手")
        payload = json.loads(repair_json_text(raw))
        if not isinstance(payload, dict):
            return {**fallback, "_source": "deterministic_invalid_llm"}
        narrative = _normalize_narrative(payload, fallback)
        return {**narrative, "_source": "llm" if narrative is not fallback else "deterministic_rejected_llm"}
    except Exception:
        return {**fallback, "_source": "deterministic_error"}
