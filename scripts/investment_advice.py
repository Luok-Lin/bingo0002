from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import akshare as ak
import pandas as pd

from agents.roles import (
    FundamentalNewsAnalyst,
    GameReferee,
    RiskManager,
    TechnicalFlowAnalyst,
    TraderAgent,
)
from dl.predictor import DLEngine
from main import _date_to_int, fetch_external_knowledge
from memory.memory_bank import MemoryBank
from rag.retriever import SimpleRAG


FEATURE_COLUMNS = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
WEB_INDEX_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "monitoring", "web_index", "dashboard_summary.json")
STABLE_CACHE_DIR = os.path.join(BASE_DIR, "data", "investment_advice", ".stable_cache")
MAX_GENERATION_RETRY = max(1, min(5, int(str(os.getenv("ADVICE_GENERATION_RETRY", "3")).strip() or "3")))
STABLE_CACHE_MINUTES = max(5, min(24 * 60, int(str(os.getenv("STABLE_CACHE_MINUTES", "120")).strip() or "120")))


def _normalize_action(value: str) -> str:
    token = str(value or "").upper().strip()
    if "BUY" in token:
        return "BUY"
    if "SELL" in token:
        return "SELL"
    return "HOLD"


def _stable_cache_path(ticker: str, as_of_date: str) -> str:
    safe_date = str(as_of_date or "").replace("-", "")
    return os.path.join(STABLE_CACHE_DIR, f"{str(ticker).zfill(6)}_{safe_date}.json")


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
    parse_fail_count = sum(1 for t in texts if "解析失败" in t)
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
            "rule_fallback_count": fallback_count,
            "empty_reason_count": empty_reason_count,
            "consensus_tech_fund": tech_sent == fund_sent,
            "referee_action": ref_decision,
            "risk_action": risk_action,
            "recommendation_action": rec_action,
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
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured investment advice for a given stock.")
    parser.add_argument("ticker", help="A股股票代码，例如 600519")
    parser.add_argument("--debate-depth", type=int, default=2, help="裁判与两个综合分析师最大博弈轮数")
    parser.add_argument("--output-path", default=None, help="结构化建议 JSON 输出路径")
    parser.add_argument("--human-comment", default="", help="人工评论，将作为经验沉淀写入记忆库")
    parser.add_argument("--human-decision", default=None, help="人工修正最终方向，可选 BUY/SELL/HOLD")
    return parser.parse_args()


def fetch_history(ticker: str) -> pd.DataFrame:
    def _finalize(df_raw: pd.DataFrame) -> pd.DataFrame:
        df_hist = df_raw.copy()
        df_hist["日期"] = pd.to_datetime(df_hist["日期"]).dt.strftime("%Y-%m-%d")
        df_hist["前收盘"] = df_hist["收盘"].shift(1)
        df_hist["涨跌额"] = df_hist["收盘"] - df_hist["前收盘"]
        df_hist["涨跌幅"] = df_hist["涨跌额"] / df_hist["前收盘"] * 100
        df_hist["振幅"] = (df_hist["最高"] - df_hist["最低"]) / df_hist["前收盘"] * 100
        df_hist["换手率"] = df_hist["换手率"] * 100
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
        return _finalize(pd.DataFrame(synth))

    prefix = "sh" if ticker.startswith("6") else "sz"
    prefixed = f"{prefix}{ticker}"
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
        df_hist = _finalize(df_hist)
    except Exception as exc:
        print(f"[WARN] 在线行情获取失败({ticker})，启用本地兜底: {exc}")
        df_hist = _fallback_from_local_backtests()

    if df_hist.empty or len(df_hist) < 20:
        raise RuntimeError(f"{ticker} 历史行情不足，无法生成结构化建议")
    return df_hist


def build_rag(ticker: str, cutoff_date: str | None = None) -> SimpleRAG:
    knowledge = fetch_external_knowledge(ticker, cutoff_date=cutoff_date)
    if not knowledge:
        knowledge = [
            {
                "page_content": f"{ticker} 暂无可用新闻研报，建议降低新闻面权重。",
                "metadata": {"date_int": _date_to_int(cutoff_date) if cutoff_date else 20991231, "ticker": ticker, "source": "fallback"},
            }
        ]
    return SimpleRAG(data_sources=knowledge)


def generate_advice(ticker: str, debate_depth: int, human_comment: str = "", human_decision: str | None = None) -> dict:
    ticker = str(ticker).strip().zfill(6)
    memory_bank = MemoryBank()
    df_hist = fetch_history(ticker)
    target_date = str(df_hist.iloc[-1]["日期"])
    if not human_comment and not human_decision:
        cached = _load_stable_cache(ticker, target_date)
        if cached:
            cache_errors = _validate_advice_payload(cached)
            if not cache_errors:
                cached = dict(cached)
                cached["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cached["cache_hit"] = True
                return cached
    features = df_hist.tail(10)[FEATURE_COLUMNS].values
    latest = df_hist.iloc[-1]

    rag_engine = build_rag(ticker, cutoff_date=target_date)
    dl_engine = DLEngine() if str(os.getenv("ENABLE_DL_ANALYST", "1")).strip() != "0" else None
    technical_flow_analyst = TechnicalFlowAnalyst(name="技术资金综合分析师", dl_engine=dl_engine)
    fundamental_news_analyst = FundamentalNewsAnalyst(name="基本面新闻综合分析师", rag_engine=rag_engine)
    referee = GameReferee(name="无情裁判官", memory_bank=memory_bank)
    risk_manager = RiskManager(name="风控大脑", memory_bank=memory_bank)
    trader = TraderAgent(name="结构化建议执行器")

    technical_case = technical_flow_analyst.step(ticker, features=features, target_date=target_date)
    fundamental_case = fundamental_news_analyst.step(ticker, target_date=target_date)
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
    final_instruction = risk_manager.step(ticker, referee_decision)
    execution_action = trader.step(final_instruction)

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
    }
    advice["cache_hit"] = False
    stability = _compute_stability(advice)
    advice.update(stability)
    advice["recommendation"]["stability_note"] = stability["stability_note"]
    return advice


def generate_advice_with_retry(
    ticker: str,
    debate_depth: int,
    human_comment: str = "",
    human_decision: str | None = None,
    max_retries: int = MAX_GENERATION_RETRY,
) -> dict:
    last_errors: list[str] = []
    for attempt in range(1, max_retries + 1):
        advice = generate_advice(
            ticker=ticker,
            debate_depth=debate_depth,
            human_comment=human_comment,
            human_decision=human_decision,
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
    print(f"\n结构化投资建议已保存: {output_path}")


if __name__ == "__main__":
    main()
