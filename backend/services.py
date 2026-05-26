from __future__ import annotations

import csv
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from typing import Any

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass

urllib.request.getproxies = lambda: {}


def _env_int(name: str, default: int, lower: int = 1) -> int:
    try:
        return max(lower, int(str(os.getenv(name, str(default))).strip() or str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


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

from agents.llm_client import LLMClient, repair_json_text
from rl.reward import compute_trade_reward

from .config import (
    ADVICE_QA_HISTORY_PATH,
    ADVICE_DIR,
    ADVICE_SETTLEMENT_PATH,
    BACKTEST_SUMMARY_DIR,
    EVOLUTION_HISTORY_PATH,
    REFLECTIONS_PATH,
    TOP_HOLDINGS_CSV,
    WATCHLIST_PATH,
)
from .indexer import build_dashboard_summary
from .strategy_rules import (
    apply_strategy_rules_to_advice,
    get_applicable_strategy_rules,
    load_strategy_rules,
    update_strategy_rules_from_records,
)
from .storage import read_json, write_json

ADVICE_EVALUATION_HORIZONS = (1, 3, 5, 10, 20)


def _run_command(
    cmd: list[str],
    cwd: str,
    *,
    detach: bool = False,
    timeout_seconds: int | None = None,
    progress_callback=None,
) -> dict:
    """Run a subprocess; detach=True keeps advice jobs alive across API reload."""
    if timeout_seconds is None:
        timeout_seconds = _env_int("TASK_SUBPROCESS_TIMEOUT_SECONDS", 3600, lower=30)
    else:
        timeout_seconds = max(30, int(timeout_seconds))
    popen_kwargs: dict = {
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if detach and os.name != "nt":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **popen_kwargs)
    timed_out = False
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    if progress_callback is not None:
        lines: queue.Queue[tuple[str, str | None]] = queue.Queue()

        def _reader(stream, name: str) -> None:
            try:
                for line in iter(stream.readline, ""):
                    lines.put((name, line))
            finally:
                try:
                    stream.close()
                except Exception:
                    pass
                lines.put((name, None))

        readers = [
            threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True),
            threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True),
        ]
        for reader in readers:
            reader.start()
        deadline = time.time() + timeout_seconds
        closed_streams: set[str] = set()
        while True:
            if time.time() > deadline and proc.poll() is None:
                timed_out = True
                if os.name != "nt":
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except Exception:
                        proc.kill()
                else:
                    proc.kill()
            try:
                name, line = lines.get(timeout=0.2)
            except queue.Empty:
                if proc.poll() is not None and len(closed_streams) >= 2:
                    break
                continue
            if line is None:
                closed_streams.add(name)
                if proc.poll() is not None and len(closed_streams) >= 2:
                    break
                continue
            if name == "stdout":
                stdout_parts.append(line)
                stripped = line.strip()
                if stripped.startswith("TA_PROGRESS "):
                    try:
                        event = json.loads(stripped.split(" ", 1)[1])
                        if isinstance(event, dict):
                            progress_callback(event)
                    except Exception:
                        pass
            else:
                stderr_parts.append(line)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        return {
            "returncode": 124 if timed_out else proc.returncode,
            "stdout": (stdout or "")[-5000:],
            "stderr": ((stderr or "") + (f"\n[TIMEOUT] command exceeded {timeout_seconds}s" if timed_out else ""))[-5000:],
            "timed_out": timed_out,
        }
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.kill()
        else:
            proc.kill()
        stdout, stderr = proc.communicate()
    return {
        "returncode": 124 if timed_out else proc.returncode,
        "stdout": (stdout or "")[-5000:],
        "stderr": ((stderr or "") + (f"\n[TIMEOUT] command exceeded {timeout_seconds}s" if timed_out else ""))[-5000:],
        "timed_out": timed_out,
    }


def _validate_advice_payload(payload: dict, *, output_path: str) -> dict:
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"建议结果为空，未生成有效 JSON：{output_path}")
    recommendation = payload.get("recommendation") or {}
    action = str(recommendation.get("action", "")).strip()
    if not action:
        raise RuntimeError(f"建议结果缺少 recommendation.action：{output_path}")
    return payload


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


def _extract_ticker_from_csv_row(row: dict) -> str:
    for key in ("股票代码", "ticker", "code", "symbol", "证券代码"):
        code = str(row.get(key, "") or "").strip()
        if code:
            digits = "".join(ch for ch in code if ch.isdigit())
            if len(digits) >= 6:
                return digits[-6:].zfill(6)
    for value in row.values():
        text = str(value or "").strip()
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 6:
            return digits[-6:].zfill(6)
    return ""


def _load_tickers_from_csv(csv_path: str, top_n: int | None = None) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            code = _extract_ticker_from_csv_row(row)
            if code and code not in seen:
                seen.add(code)
                tickers.append(code)
                if top_n and top_n > 0 and len(tickers) >= top_n:
                    break
    return tickers


def _load_top_tickers(csv_path: str, top_n: int) -> list[str]:
    return _load_tickers_from_csv(csv_path, top_n=top_n)


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
    user_profile: dict | None = None,
    progress_callback=None,
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
    personalization_preferences = (
        user_profile.get("personalization_preferences")
        if isinstance(user_profile, dict) and isinstance(user_profile.get("personalization_preferences"), dict)
        else {}
    )
    normalized_preferences = _normalize_personalization_preferences(personalization_preferences)
    if (
        not normalized_preferences["use_system_rules"]
        or normalized_preferences["selected_rule_ids"]
        or normalized_preferences["disabled_rule_ids"]
    ):
        cmd.append("--skip-strategy-rules")
    advice_timeout = _env_int("ADVICE_SUBPROCESS_TIMEOUT_SECONDS", 1800, lower=60)

    def _handle_progress(event: dict) -> None:
        if not callable(progress_callback) or not isinstance(event, dict):
            return
        progress_callback(
            stage=str(event.get("stage", "")),
            progress_percent=event.get("progress_percent", event.get("percent")),
            message=str(event.get("summary") or event.get("message") or ""),
            event=event,
            partial_result=event.get("partial_result") if isinstance(event.get("partial_result"), dict) else None,
        )

    run_result = _run_command(
        cmd,
        cwd=base_dir,
        detach=True,
        timeout_seconds=advice_timeout,
        progress_callback=_handle_progress if callable(progress_callback) else None,
    )
    returncode = int(run_result.get("returncode", 1))
    if returncode != 0:
        if os.path.exists(output_path):
            try:
                advice = _validate_advice_payload(read_json(output_path, {}), output_path=output_path)
                advice = _apply_strategy_rules_with_preferences(advice, user_profile)
                advice.setdefault("warnings", [])
                if isinstance(advice["warnings"], list):
                    advice["warnings"].append(f"建议脚本退出码 {returncode}，但已回收到输出文件。")
                if user_profile:
                    advice = _apply_user_profile_to_advice(advice, user_profile)
                write_json(output_path, advice)
                return {
                    "command": cmd,
                    "run_result": run_result,
                    "output_path": output_path,
                    "advice": advice,
                }
            except Exception:
                pass
        err_tail = str(run_result.get("stderr") or run_result.get("stdout") or "").strip()
        hint = "（若后端刚重启，请重新点击生成建议）" if returncode in {-15, -9, -2} else ""
        raise RuntimeError(
            f"建议脚本执行失败，退出码 {returncode}{hint}"
            + (f"：{err_tail[-400:]}" if err_tail else "")
        )
    if not os.path.exists(output_path):
        raise RuntimeError(f"建议脚本未写出结果文件：{output_path}")
    advice = _validate_advice_payload(read_json(output_path, {}), output_path=output_path)
    advice = _apply_strategy_rules_with_preferences(advice, user_profile)
    if user_profile:
        advice = _apply_user_profile_to_advice(advice, user_profile)
    write_json(output_path, advice)

    def _refresh_dashboard() -> None:
        try:
            build_dashboard_summary()
        except Exception:
            pass

    threading.Thread(target=_refresh_dashboard, daemon=True).start()
    return {
        "command": cmd,
        "run_result": run_result,
        "output_path": output_path,
        "advice": advice,
        "dashboard_updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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


def get_advice_for_report(
    ticker: str,
    *,
    advice_id: str | None = None,
    generated_at: str | None = None,
) -> tuple[dict, str]:
    ticker = str(ticker).strip().zfill(6)
    requested_id = str(advice_id or "").strip()
    requested_generated_at = str(generated_at or "").strip()
    if requested_id and not re.match(rf"^{re.escape(ticker)}_\d{{8}}_\d{{6}}$", requested_id):
        raise FileNotFoundError(f"未找到 {ticker} 对应的建议报告。")
    if not os.path.isdir(ADVICE_DIR):
        raise FileNotFoundError(f"未找到 {ticker} 的建议文件。")

    candidates: list[tuple[str, float, str, dict]] = []
    for name in os.listdir(ADVICE_DIR):
        if not (name.startswith(f"{ticker}_") and name.endswith(".json")):
            continue
        file_id = os.path.splitext(name)[0]
        if requested_id and file_id != requested_id:
            continue
        full = os.path.join(ADVICE_DIR, name)
        payload = read_json(full, {})
        if not isinstance(payload, dict):
            continue
        payload_ticker = str(payload.get("ticker") or ticker).strip().zfill(6)
        if payload_ticker != ticker:
            continue
        if requested_generated_at and str(payload.get("generated_at") or "").strip() != requested_generated_at:
            continue
        generated_key = str(payload.get("generated_at") or "")
        candidates.append((generated_key, os.path.getmtime(full), file_id, payload))

    if not candidates:
        raise FileNotFoundError(f"未找到 {ticker} 对应的建议报告。")
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    _, _, file_id, payload = candidates[0]
    return payload, file_id


def _clip_text(value: Any, limit: int = 260) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _append_unique_evidence(items: list[dict], *, source: str, text: Any, meta: str = "") -> None:
    clean = _clip_text(text, 320)
    if not clean:
        return
    key = clean.lower()
    if any(str(item.get("text", "")).lower() == key for item in items):
        return
    items.append({"source": source, "text": clean, "meta": _clip_text(meta, 80)})


def _collect_advice_evidence(advice: dict) -> list[dict]:
    evidence: list[dict] = []
    rec = advice.get("recommendation", {}) or {}
    risk = advice.get("risk", {}) or {}
    referee = advice.get("referee", {}) or {}
    analysts = advice.get("analyst_cases", {}) or {}
    technical = analysts.get("technical_flow", {}) or {}
    fundamental = analysts.get("fundamental_news", {}) or {}
    data_quality = advice.get("data_quality", {}) or {}
    technical_features = advice.get("technical_features", {}) or {}

    _append_unique_evidence(evidence, source="最终建议", text=rec.get("reason"), meta=rec.get("action"))
    _append_unique_evidence(evidence, source="风控", text=risk.get("reason"), meta=risk.get("decision") or risk.get("action"))
    _append_unique_evidence(evidence, source="技术分析师", text=technical.get("reasoning"), meta=technical.get("sentiment"))
    _append_unique_evidence(evidence, source="基本面新闻分析师", text=fundamental.get("reasoning"), meta=fundamental.get("sentiment"))
    _append_unique_evidence(evidence, source="裁判", text=referee.get("reason"), meta=referee.get("decision"))
    _append_unique_evidence(evidence, source="数据质量", text=data_quality.get("note"), meta=data_quality.get("level"))
    signals = technical_features.get("signals", []) if isinstance(technical_features, dict) else []
    if signals:
        _append_unique_evidence(evidence, source="技术特征", text=" ".join(str(x) for x in signals[:5]), meta=technical_features.get("version", ""))

    for case_name, case in analysts.items():
        if not isinstance(case, dict):
            continue
        for report in case.get("source_reports", []) or []:
            if not isinstance(report, dict):
                continue
            report_text = report.get("reasoning") or report.get("thought_process")
            agent = report.get("agent") or case_name
            _append_unique_evidence(evidence, source=str(agent), text=report_text, meta=report.get("sentiment"))

    for item in referee.get("debate_trace", []) or []:
        if not isinstance(item, dict):
            continue
        judge = item.get("judge", {}) or {}
        _append_unique_evidence(
            evidence,
            source=f"博弈轮次R{item.get('round', '-')}",
            text=judge.get("reasoning") or judge.get("reason"),
            meta=judge.get("decision") or judge.get("sentiment"),
        )

    return evidence[:12]


def _build_advice_qa_context(advice: dict, evidence: list[dict]) -> str:
    rec = advice.get("recommendation", {}) or {}
    risk = advice.get("risk", {}) or {}
    referee = advice.get("referee", {}) or {}
    multi_period = advice.get("multi_period_advice", {}) or {}
    short_term = multi_period.get("short_term") or advice.get("short_term") or {}
    swing_term = multi_period.get("swing_term") or advice.get("swing_term") or {}
    risk_plan = multi_period.get("risk_plan") or advice.get("risk_plan") or {}
    latest = advice.get("latest_market", {}) or {}
    quality = advice.get("data_quality", {}) or {}
    stability = advice.get("stability_diagnostics", {}) or {}
    technical_features = advice.get("technical_features", {}) or {}
    trend = technical_features.get("trend", {}) if isinstance(technical_features, dict) else {}
    momentum = technical_features.get("momentum", {}) if isinstance(technical_features, dict) else {}
    risk_features = technical_features.get("risk", {}) if isinstance(technical_features, dict) else {}
    volume_features = technical_features.get("volume", {}) if isinstance(technical_features, dict) else {}
    evidence_lines = [
        f"- [{item.get('source', '证据')}] {item.get('text', '')}"
        for item in evidence
        if item.get("text")
    ]
    return "\n".join(
        [
            f"标的: {advice.get('ticker', '')}",
            f"建议日期: {advice.get('as_of_date', '')}; 生成时间: {advice.get('generated_at', '')}",
            (
                "最新行情: "
                f"收盘={latest.get('close', '-')}, 涨跌幅={latest.get('pct_change', '-')}%, "
                f"换手率={latest.get('turnover', '-')}"
            ),
            (
                "最终建议: "
                f"action={rec.get('action', '-')}, execution={rec.get('execution_action', '-')}, "
                f"position={rec.get('position_percent', '-')}%, confidence={rec.get('confidence', '-')}"
            ),
            f"建议理由: {_clip_text(rec.get('reason'), 700)}",
            (
                "裁判: "
                f"decision={referee.get('decision', '-')}, bull_score={referee.get('bull_score', '-')}, "
                f"bear_score={referee.get('bear_score', '-')}, trend_strength={referee.get('trend_strength', '-')}"
            ),
            (
                "多周期建议: "
                f"short_term={short_term.get('action', '-')}({short_term.get('confidence', '-')}), "
                f"reason={_clip_text(short_term.get('reason'), 260)}; "
                f"swing_term={swing_term.get('action', '-')}({swing_term.get('confidence', '-')}), "
                f"reason={_clip_text(swing_term.get('reason'), 260)}"
            ),
            (
                "风控计划: "
                f"stop_loss={_clip_text(risk_plan.get('stop_loss'), 220)}, "
                f"take_profit={_clip_text(risk_plan.get('take_profit'), 220)}, "
                f"invalid={', '.join(str(x) for x in (risk_plan.get('invalid_conditions') or [])[:5])}"
            ),
            (
                "风控: "
                f"decision={risk.get('decision') or risk.get('action') or '-'}, "
                f"position={risk.get('position_percent', rec.get('position_percent', '-'))}%, "
                f"reason={_clip_text(risk.get('reason'), 500)}"
            ),
            (
                "数据质量: "
                f"score={quality.get('score', '-')}, level={quality.get('level', '-')}, "
                f"note={_clip_text(quality.get('note'), 400)}"
            ),
            (
                "技术特征: "
                f"MA20偏离={trend.get('price_vs_ma20_pct', '-')}%, "
                f"MA20斜率={trend.get('ma20_slope_5d_pct', '-')}%, "
                f"RSI14={momentum.get('rsi14', '-')}, MACD柱={momentum.get('macd_hist', '-')}, "
                f"ATR14占比={risk_features.get('atr14_pct', '-')}%, "
                f"量比={volume_features.get('volume_ratio_5_20', '-')}"
            ),
            (
                "稳定性诊断: "
                f"parse_fail={stability.get('parse_fail_count', '-')}, "
                f"rule_fallback={stability.get('rule_fallback_count', '-')}, "
                f"empty_reason={stability.get('empty_reason_count', '-')}"
            ),
            "可引用证据:\n" + ("\n".join(evidence_lines) if evidence_lines else "- 暂无可引用证据"),
        ]
    )


def _fallback_qa_answer(question: str, advice: dict, evidence: list[dict], reason: str = "") -> dict:
    rec = advice.get("recommendation", {}) or {}
    quality = advice.get("data_quality", {}) or {}
    multi_period = advice.get("multi_period_advice", {}) or {}
    short_term = multi_period.get("short_term") or advice.get("short_term") or {}
    swing_term = multi_period.get("swing_term") or advice.get("swing_term") or {}
    action = _normalize_action(rec.get("action", "HOLD"))
    position = _safe_float(rec.get("position_percent"), 0.0)
    core_reason = _clip_text(rec.get("reason"), 180) or "当前建议文件没有给出完整理由。"
    quality_note = _clip_text(quality.get("note"), 160)
    answer = (
        f"围绕你的问题“{_clip_text(question, 80)}”，最新结构化建议给出的动作是 {action}，"
        f"建议仓位约 {position:.1f}%。核心依据是：{core_reason}"
    )
    if short_term or swing_term:
        answer += (
            f" 多周期拆分看，短线为 {short_term.get('action', '-')}（{short_term.get('horizon', '1-3 trading days')}），"
            f"中线为 {swing_term.get('action', '-')}（{swing_term.get('horizon', '2-4 weeks')}）。"
        )
    if quality_note:
        answer += f" 数据质量提示：{quality_note}"
    if reason:
        answer += f" 本次回答使用本地结构化建议兜底生成，原因：{_clip_text(reason, 120)}"
    return {
        "ticker": advice.get("ticker", ""),
        "question": question,
        "answer": answer,
        "evidence": evidence[:5],
        "confidence": 0.42 if reason else 0.55,
        "follow_up_questions": [
            "这个结论在什么条件下会改变？",
            "当前最大风险是什么？",
            "如果我已经持仓，仓位该怎么处理？",
        ],
        "limitations": "该回答基于最新建议文件和可用证据生成，不构成投资承诺。",
        "llm_fallback": bool(reason),
    }


def _parse_qa_llm_response(raw: str, question: str, advice: dict, evidence: list[dict]) -> dict:
    try:
        parsed = json.loads(repair_json_text(raw))
    except Exception as exc:
        return _fallback_qa_answer(question, advice, evidence, reason=f"LLM JSON 解析失败: {exc}")
    if not isinstance(parsed, dict) or not str(parsed.get("answer", "")).strip():
        return _fallback_qa_answer(question, advice, evidence, reason="LLM 未返回 answer 字段")

    llm_evidence = parsed.get("evidence", [])
    normalized_evidence: list[dict] = []
    if isinstance(llm_evidence, list):
        for item in llm_evidence[:6]:
            if isinstance(item, dict):
                _append_unique_evidence(
                    normalized_evidence,
                    source=str(item.get("source", "证据")),
                    text=item.get("text") or item.get("quote") or item.get("summary"),
                    meta=item.get("meta", ""),
                )
            else:
                _append_unique_evidence(normalized_evidence, source="证据", text=item)
    if not normalized_evidence:
        normalized_evidence = evidence[:5]

    follow_ups = parsed.get("follow_up_questions", [])
    if not isinstance(follow_ups, list):
        follow_ups = []
    follow_ups = [_clip_text(x, 80) for x in follow_ups if str(x or "").strip()][:4]
    if not follow_ups:
        follow_ups = _fallback_qa_answer(question, advice, evidence)["follow_up_questions"]

    return {
        "ticker": advice.get("ticker", ""),
        "question": question,
        "answer": _clip_text(parsed.get("answer"), 1800),
        "evidence": normalized_evidence[:6],
        "confidence": max(0.0, min(1.0, _safe_float(parsed.get("confidence"), 0.6))),
        "follow_up_questions": follow_ups,
        "limitations": _clip_text(parsed.get("limitations"), 240) or "回答基于当前系统已有建议与检索证据生成。",
        "llm_fallback": False,
    }


def _qa_llm_client() -> LLMClient:
    config = LLMClient.from_env()
    return LLMClient(
        replace(
            config,
            timeout_seconds=_env_int("ADVICE_QA_LLM_TIMEOUT_SECONDS", 25, lower=5),
            json_retry=min(_env_int("ADVICE_QA_LLM_JSON_RETRY", 1, lower=1), 3),
            max_tokens=_env_int("ADVICE_QA_LLM_MAX_TOKENS", 700, lower=128),
            response_format_json=_env_bool("ADVICE_QA_LLM_RESPONSE_FORMAT_JSON", True),
        )
    )


def _load_qa_history() -> list[dict]:
    history = read_json(ADVICE_QA_HISTORY_PATH, [])
    return history if isinstance(history, list) else []


def ask_advice_question(
    *,
    ticker: str,
    question: str,
    include_latest_advice: bool = True,
    include_rag: bool = False,
    user_id: str = "",
) -> dict:
    ticker = str(ticker).strip().zfill(6)
    question = _clip_text(question, 500)
    if not ticker or not question:
        raise ValueError("股票代码和问题不能为空。")
    if not include_latest_advice:
        raise ValueError("当前问答助手需要基于最新结构化建议回答，请开启 include_latest_advice。")

    advice = get_latest_advice_for_ticker(ticker)
    if not advice:
        raise FileNotFoundError(f"未找到 {ticker} 的最新建议，请先生成建议。")

    evidence = _collect_advice_evidence(advice)
    rag_requested = bool(include_rag)
    rag_allowed = _env_bool("ADVICE_QA_ENABLE_RAG", False)
    rag_status = {
        "enabled": rag_requested and rag_allowed,
        "requested": rag_requested,
        "documents": 0,
        "status": "skipped",
        "error": "",
    }
    if rag_requested and not rag_allowed:
        rag_status.update(
            {
                "status": "disabled_by_env",
                "error": "追问默认使用最新建议快答；如需每次追问重检索，设置 ADVICE_QA_ENABLE_RAG=1。",
            }
        )
    if rag_requested and rag_allowed:
        try:
            from scripts.investment_advice import build_rag_with_diagnostics

            rag_engine, rag_diagnostics = build_rag_with_diagnostics(ticker, cutoff_date=str(advice.get("as_of_date", "")))
            rag_docs = rag_engine.retrieve(question, target_date=str(advice.get("as_of_date", "")), ticker=ticker, top_k=4)
            for doc in rag_docs:
                _append_unique_evidence(evidence, source="RAG检索", text=doc, meta="news/report")
            rag_status = {
                "enabled": True,
                "requested": True,
                "documents": len(rag_docs),
                "status": rag_diagnostics.get("status", "ok"),
                "error": rag_diagnostics.get("degraded_reason") or "; ".join(rag_diagnostics.get("errors", []) or []),
            }
        except Exception as exc:
            rag_status = {"enabled": True, "requested": True, "documents": 0, "status": "error", "error": str(exc)[:240]}

    context = _build_advice_qa_context(advice, evidence)
    prompt = f"""
你是一个谨慎的 A 股投研问答助手。请只基于【最新结构化建议】和【可引用证据】回答，不要编造不存在的数据。
如果证据不足，请明确说明哪些信息不足，并给出可操作的复核方向。

【用户问题】
{question}

【最新结构化建议上下文】
{context}

【输出格式】只输出一个合法 JSON 对象，不要包含 Markdown 或额外文字。字段固定为：
{{
  "answer": "直接回答用户问题，120到260字，包含动作、理由、风险或条件",
  "evidence": [{{"source":"证据来源","text":"用于支撑回答的简短证据"}}],
  "confidence": 0.0,
  "follow_up_questions": ["自然的后续追问1","自然的后续追问2","自然的后续追问3"],
  "limitations": "一句话说明边界"
}}
"""
    use_llm = _env_bool("ADVICE_QA_USE_LLM", True)
    if use_llm:
        try:
            raw = _qa_llm_client().query(prompt, role="投研问答助手")
            response = _parse_qa_llm_response(raw, question, advice, evidence)
        except Exception as exc:
            response = _fallback_qa_answer(question, advice, evidence, reason=f"LLM 请求异常: {exc}")
        answer_mode = "llm_fallback" if response.get("llm_fallback") else "llm"
    else:
        response = _fallback_qa_answer(question, advice, evidence)
        response["llm_fallback"] = False
        answer_mode = "local_fast"
    response.update(
        {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "as_of_date": advice.get("as_of_date", ""),
            "advice_generated_at": advice.get("generated_at", ""),
            "answer_mode": answer_mode,
            "rag_status": rag_status,
        }
    )

    history = _load_qa_history()
    history.append(
        {
            "user_id": user_id,
            "ticker": ticker,
            "question": question,
            "answer": response.get("answer", ""),
            "confidence": response.get("confidence"),
            "generated_at": response["generated_at"],
            "as_of_date": response.get("as_of_date", ""),
        }
    )
    write_json(ADVICE_QA_HISTORY_PATH, history[-1000:])
    return response


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


def resolve_evolution_tickers(
    tickers: list[str] | None = None,
    csv_path: str | None = None,
    csv_top_n: int | None = None,
) -> list[str]:
    if tickers:
        normalized = [str(t).strip().zfill(6) for t in tickers if str(t).strip()]
        if normalized:
            return list(dict.fromkeys(normalized))
    if csv_path:
        loaded = _load_tickers_from_csv(csv_path, top_n=csv_top_n)
        if not loaded:
            raise ValueError(f"CSV 未解析到有效股票代码: {csv_path}")
        return loaded

    watchlist = read_json(WATCHLIST_PATH, {})
    fallback = [str(t).zfill(6) for t in watchlist.get("tickers", []) if str(t).strip()]
    if fallback:
        return fallback
    return _load_top_tickers(TOP_HOLDINGS_CSV, 10)


def run_daily_evolution(
    base_dir: str,
    tickers: list[str] | None = None,
    debate_depth: int = 2,
    mode: str = "backtest_update",
    csv_path: str | None = None,
    csv_top_n: int | None = None,
) -> dict:
    tickers = resolve_evolution_tickers(tickers=tickers, csv_path=csv_path, csv_top_n=csv_top_n)
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


def _normalize_user_profile(profile: dict | None) -> dict:
    raw = profile if isinstance(profile, dict) else {}
    def _safe_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        token = str(value if value is not None else "").strip().lower()
        if token in {"1", "true", "yes", "on", "y"}:
            return True
        if token in {"0", "false", "no", "off", "n"}:
            return False
        return default

    risk_profile = str(raw.get("risk_profile", "balanced") or "balanced").strip().lower()
    if risk_profile not in {"conservative", "balanced", "aggressive"}:
        risk_profile = "balanced"
    holding_period = str(raw.get("holding_period", "swing") or "swing").strip().lower()
    if holding_period not in {"intraday", "swing", "mid_term"}:
        holding_period = "swing"
    max_position_per_stock = max(1.0, min(100.0, _safe_float(raw.get("max_position_per_stock"), 20.0)))
    already_holding = _safe_bool(raw.get("already_holding", False))
    current_position_percent = max(0.0, min(100.0, _safe_float(raw.get("current_position_percent"), 0.0)))
    cost_price = max(0.0, _safe_float(raw.get("cost_price"), 0.0))
    prefer_stop_loss = _safe_bool(raw.get("prefer_stop_loss", True), default=True)
    return {
        "risk_profile": risk_profile,
        "holding_period": holding_period,
        "max_position_per_stock": round(max_position_per_stock, 2),
        "already_holding": already_holding,
        "current_position_percent": round(current_position_percent, 2),
        "cost_price": round(cost_price, 4),
        "prefer_stop_loss": prefer_stop_loss,
    }


def _normalize_personalization_preferences(preferences: dict | None) -> dict:
    raw = preferences if isinstance(preferences, dict) else {}

    def _safe_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        token = str(value if value is not None else "").strip().lower()
        if token in {"1", "true", "yes", "on", "y"}:
            return True
        if token in {"0", "false", "no", "off", "n"}:
            return False
        return default

    def _string_list(value: Any, limit: int = 100) -> list[str]:
        values = value if isinstance(value, list) else []
        out: list[str] = []
        for item in values:
            text = str(item or "").strip()
            if text:
                out.append(text[:120])
        return list(dict.fromkeys(out))[:limit]

    def _infer_custom_rule_policy(text: str, fallback: str = "") -> str:
        explicit = str(fallback or "").strip().lower()
        if explicit in {"block_buy", "block_sell", "force_hold", "position_cap", "min_confidence"}:
            return explicit
        if re.search(r"仓位|不超过|上限|半仓|轻仓|小仓", text):
            return "position_cap"
        if re.search(r"置信|信心|确定性|把握|概率", text):
            return "min_confidence"
        if re.search(r"观望|等待|暂不|不操作|空仓|休息", text):
            return "force_hold"
        if re.search(r"卖|减仓|止盈|止损|清仓", text):
            return "block_sell"
        return "block_buy"

    def _number_from_text(text: str, default: float = 0.0) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if m:
            return _safe_float(m.group(1), default)
        m = re.search(r"0\.\d+|\d+(?:\.\d+)?", text)
        if m:
            return _safe_float(m.group(0), default)
        return default

    def _normalize_policy_items(items: Any, *, kind: str) -> list[dict[str, Any]]:
        normalized_items = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            text = _clip_text(item.get("text") or item.get("label") or "", 180)
            raw_policy = str(item.get("policy" if kind == "rule" else "type", item.get("type", "")) or "").strip().lower()
            ctype = _infer_custom_rule_policy(text, raw_policy) if kind == "rule" else raw_policy
            if ctype not in {"block_buy", "block_sell", "force_hold", "position_cap", "min_confidence"}:
                continue
            value = item.get("value")
            normalized: dict[str, Any] = {
                "text": text,
                "enabled": _safe_bool(item.get("enabled", True), default=True),
            }
            if kind == "rule":
                normalized["policy"] = ctype
            else:
                normalized["type"] = ctype
            if ctype in {"position_cap", "min_confidence"}:
                parsed = _safe_float(value, 0.0)
                if parsed <= 0:
                    parsed = _number_from_text(text, 0.0)
                if ctype == "min_confidence" and parsed > 1:
                    parsed = parsed / 100.0
                normalized["value"] = parsed
            normalized_items.append(normalized)
        return normalized_items

    constraints = _normalize_policy_items(raw.get("custom_constraints", []), kind="constraint")
    custom_rules = _normalize_policy_items(raw.get("custom_rules", []), kind="rule")

    return {
        "use_system_rules": _safe_bool(raw.get("use_system_rules", True), default=True),
        "selected_rule_ids": _string_list(raw.get("selected_rule_ids"), limit=200),
        "disabled_rule_ids": _string_list(raw.get("disabled_rule_ids"), limit=200),
        "custom_rules": custom_rules[:50],
        "custom_constraints": constraints[:50],
    }


def _strategy_rule_filter_for_preferences(preferences: dict) -> tuple[set[str] | None, set[str]]:
    normalized = _normalize_personalization_preferences(preferences)
    if not normalized["use_system_rules"]:
        return set(), set()
    disabled = set(normalized["disabled_rule_ids"])
    selected = [rule_id for rule_id in normalized["selected_rule_ids"] if rule_id not in disabled]
    if selected:
        return set(selected), disabled
    return None, disabled


def _apply_strategy_rules_with_preferences(advice: dict, user_profile: dict | None = None) -> dict:
    profile = user_profile if isinstance(user_profile, dict) else {}
    preferences = _normalize_personalization_preferences(
        profile.get("personalization_preferences") if isinstance(profile.get("personalization_preferences"), dict) else {}
    )
    allowed_rule_ids, disabled_rule_ids = _strategy_rule_filter_for_preferences(preferences)
    if allowed_rule_ids == set():
        advice["strategy_rules"] = {
            "applied": False,
            "candidate_count": len(get_applicable_strategy_rules(advice)),
            "applied_rules": [],
            "user_rule_preference": "disabled",
        }
        return advice
    return apply_strategy_rules_to_advice(
        advice,
        allowed_rule_ids=allowed_rule_ids,
        disabled_rule_ids=disabled_rule_ids,
    )


def _apply_user_profile_to_advice(advice: dict, user_profile: dict) -> dict:
    if not isinstance(advice, dict):
        return advice
    profile = _normalize_user_profile(user_profile)
    recommendation = advice.get("recommendation")
    if not isinstance(recommendation, dict):
        return advice
    risk = advice.get("risk")
    if not isinstance(risk, dict):
        risk = {}
        advice["risk"] = risk

    original_action = _normalize_action(recommendation.get("action", "HOLD"))
    action = original_action
    notes: list[str] = []

    data_quality = advice.get("data_quality") if isinstance(advice.get("data_quality"), dict) else {}
    quality_level = str(data_quality.get("level", "")).strip().lower()
    quality_score = _safe_float(data_quality.get("score"), 1.0)
    if profile["risk_profile"] == "conservative" and (quality_level == "low" or quality_score < 0.45):
        action = "HOLD"
        notes.append("保守画像触发：数据质量偏低，动作强制降级为 HOLD。")

    multi_period = advice.get("multi_period_advice") if isinstance(advice.get("multi_period_advice"), dict) else {}
    short_term = multi_period.get("short_term") if isinstance(multi_period.get("short_term"), dict) else {}
    swing_term = multi_period.get("swing_term") if isinstance(multi_period.get("swing_term"), dict) else {}
    if profile["holding_period"] == "intraday":
        period_action = _normalize_action(short_term.get("action", action))
        if period_action != action:
            action = period_action
            notes.append("持有周期为日内，执行动作对齐短线建议。")
    elif profile["holding_period"] == "mid_term":
        period_action = _normalize_action(swing_term.get("action", action))
        if period_action != action:
            action = period_action
            notes.append("持有周期为中线，执行动作对齐中线建议。")

    trend_strength = _safe_float((advice.get("referee") or {}).get("trend_strength"), 0.0)
    confidence = max(0.0, min(1.0, _safe_float(recommendation.get("confidence"), 0.0)))
    if (
        profile["risk_profile"] == "aggressive"
        and action == "HOLD"
        and trend_strength >= 0.65
        and confidence >= 0.55
        and quality_level != "low"
    ):
        action = "BUY"
        notes.append("激进画像触发：趋势强且置信度尚可，允许突破买入。")

    raw_position = max(0.0, min(100.0, _safe_float(recommendation.get("position_percent"), 0.0)))
    max_position = profile["max_position_per_stock"]
    current_position = profile["current_position_percent"] if profile["already_holding"] else 0.0
    target_position = raw_position

    if action == "BUY":
        if profile["already_holding"]:
            if current_position >= max_position:
                action = "HOLD"
                target_position = max_position
                notes.append("已持仓且接近单票仓位上限，暂停加仓，转为 HOLD。")
            else:
                target_position = min(max(raw_position, current_position), max_position)
                if target_position > current_position:
                    notes.append(f"已持仓画像：允许加仓至不超过 {max_position:.1f}%。")
        else:
            first_entry_limit = min(10.0, max_position)
            target_position = min(raw_position, first_entry_limit)
            notes.append(f"空仓画像：首次建仓仓位限制为不超过 {first_entry_limit:.1f}%。")
    elif action == "SELL":
        if profile["already_holding"]:
            target_position = max(0.0, min(current_position, raw_position if raw_position > 0 else current_position))
            notes.append("已持仓画像：SELL 解释为减仓/止盈止损动作。")
        else:
            target_position = 0.0
    else:
        if profile["already_holding"]:
            target_position = min(current_position, max_position)
            notes.append("已持仓画像：HOLD 维持当前仓位管理。")
        else:
            target_position = 0.0

    if action == "BUY" and target_position <= 0:
        action = "HOLD"
        target_position = 0.0
        notes.append("买入方向被仓位约束压至 0%，执行层转为 HOLD。")
    elif action == "SELL" and not profile["already_holding"] and target_position <= 0:
        action = "HOLD"
        target_position = 0.0
        notes.append("空仓状态无可卖持仓，执行层转为 HOLD。")

    preferences = _normalize_personalization_preferences(
        user_profile.get("personalization_preferences") if isinstance(user_profile.get("personalization_preferences"), dict) else {}
    )
    custom_rules = [item for item in preferences.get("custom_rules", []) if item.get("enabled")]
    custom_constraints = [item for item in preferences.get("custom_constraints", []) if item.get("enabled")]
    custom_notes: list[str] = []
    actionable_items = [
        {"source": "用户自定义策略", "policy": item.get("policy"), **item}
        for item in custom_rules
    ] + [
        {"source": "自定义约束", "policy": item.get("type"), **item}
        for item in custom_constraints
    ]
    for constraint in actionable_items:
        ctype = str(constraint.get("policy", "") or "")
        source = str(constraint.get("source") or "自定义约束")
        label = _clip_text(constraint.get("text"), 120)
        suffix = f"：{label}" if label else ""
        if ctype == "force_hold" and action != "HOLD":
            action = "HOLD"
            target_position = current_position if profile["already_holding"] else 0.0
            custom_notes.append(f"{source}强制观望{suffix}")
        elif ctype == "block_buy" and action == "BUY":
            action = "HOLD"
            target_position = current_position if profile["already_holding"] else 0.0
            custom_notes.append(f"{source}禁止买入{suffix}")
        elif ctype == "block_sell" and action == "SELL":
            action = "HOLD"
            target_position = current_position if profile["already_holding"] else 0.0
            custom_notes.append(f"{source}禁止卖出{suffix}")
        elif ctype == "position_cap":
            cap = max(0.0, min(100.0, _safe_float(constraint.get("value"), target_position)))
            if target_position > cap:
                target_position = cap
                custom_notes.append(f"{source}仓位上限 {cap:.1f}%{suffix}")
        elif ctype == "min_confidence":
            threshold = max(0.0, min(1.0, _safe_float(constraint.get("value"), 0.0)))
            if action in {"BUY", "SELL"} and confidence < threshold:
                action = "HOLD"
                target_position = current_position if profile["already_holding"] else 0.0
                custom_notes.append(f"置信度 {confidence:.2f} 低于{source}阈值 {threshold:.2f}{suffix}")
    if action in {"BUY", "SELL"} and target_position <= 0:
        action = "HOLD"
        target_position = 0.0
        custom_notes.append("自定义约束后方向性仓位为 0%，执行层转为 HOLD。")
    notes.extend(custom_notes)

    recommendation["action"] = action
    recommendation["execution_action"] = action if action == "HOLD" else f"{action} {target_position:.1f}%"
    recommendation["position_percent"] = round(target_position, 2)

    existing_reason = str(recommendation.get("reason", "") or "").strip()
    if notes:
        recommendation["reason"] = (
            f"{existing_reason}｜用户画像约束：{'；'.join(notes)}" if existing_reason else f"用户画像约束：{'；'.join(notes)}"
        )
    recommendation["personalized"] = True

    risk_reason = str(risk.get("reason", "") or "").strip()
    if profile["prefer_stop_loss"] and profile["already_holding"] and profile["cost_price"] > 0:
        stop_ratio = {"conservative": 0.95, "balanced": 0.93, "aggressive": 0.9}[profile["risk_profile"]]
        stop_price = profile["cost_price"] * stop_ratio
        stop_note = f"持仓止损参考价 {stop_price:.2f}（成本 {profile['cost_price']:.2f}）。"
        risk["stop_loss"] = (
            f"{str(risk.get('stop_loss', '')).strip()}；{stop_note}".strip("；")
            if str(risk.get("stop_loss", "")).strip()
            else stop_note
        )
        if stop_note not in risk_reason:
            risk["reason"] = f"{risk_reason}；{stop_note}".strip("；")
        risk_plan = advice.get("risk_plan") if isinstance(advice.get("risk_plan"), dict) else {}
        multi_period = advice.get("multi_period_advice") if isinstance(advice.get("multi_period_advice"), dict) else {}
        multi_risk_plan = multi_period.get("risk_plan") if isinstance(multi_period.get("risk_plan"), dict) else {}
        for plan in (risk_plan, multi_risk_plan):
            existing_stop = str(plan.get("stop_loss", "") or "").strip()
            plan["stop_loss"] = f"{existing_stop}；{stop_note}".strip("；") if existing_stop else stop_note
        if risk_plan:
            advice["risk_plan"] = risk_plan
        if multi_period and multi_risk_plan:
            multi_period["risk_plan"] = multi_risk_plan
            advice["multi_period_advice"] = multi_period
    risk["position_percent"] = round(target_position, 2)
    risk["final_action"] = action
    risk["action"] = action
    risk["decision"] = action

    advice["user_profile"] = profile
    advice["personalization"] = {
        "applied": True,
        "risk_profile": profile["risk_profile"],
        "holding_period": profile["holding_period"],
        "notes": notes,
        "custom_rules": custom_rules,
        "custom_constraints": custom_constraints,
        "rule_preferences": {
            "use_system_rules": preferences["use_system_rules"],
            "selected_rule_ids": preferences["selected_rule_ids"],
            "disabled_rule_ids": preferences["disabled_rule_ids"],
        },
        "scope": user_profile.get("personalization_scope") if isinstance(user_profile, dict) else "",
        "ticker": user_profile.get("personalization_ticker") if isinstance(user_profile, dict) else "",
        "original_action": original_action,
        "adjusted_action": action,
    }
    return advice


def _normalize_portfolio_holdings(holdings: list[dict]) -> list[dict]:
    normalized = []
    for raw in holdings or []:
        if not isinstance(raw, dict):
            continue
        ticker = str(raw.get("ticker", "") or raw.get("code", "") or raw.get("symbol", "")).strip()
        digits = "".join(ch for ch in ticker if ch.isdigit())
        if len(digits) >= 6:
            ticker = digits[-6:].zfill(6)
        else:
            ticker = ticker.zfill(6) if ticker else ""
        if not ticker:
            continue
        weight = raw.get("weight_percent")
        if weight is None:
            weight = raw.get("current_position_percent", raw.get("position_percent", 0.0))
        normalized.append(
            {
                "ticker": ticker,
                "weight_percent": round(max(0.0, min(100.0, _safe_float(weight, 0.0))), 2),
                "cost_price": round(max(0.0, _safe_float(raw.get("cost_price"), 0.0)), 4),
                "note": _clip_text(raw.get("note"), 120),
            }
        )
    return normalized[:100]


def _target_cash_percent(profile: dict, objective: str, requested_cash: float) -> float:
    objective = str(objective or "balanced").strip().lower()
    risk_profile = str(profile.get("risk_profile", "balanced") or "balanced").lower()
    base = {"conservative": 20.0, "balanced": 12.0, "aggressive": 6.0}.get(risk_profile, 12.0)
    if objective in {"income", "defensive", "防守", "稳健"}:
        base += 8.0
    elif objective in {"growth", "aggressive", "进攻", "成长"}:
        base -= 4.0
    elif objective in {"rebalance", "再平衡"}:
        base += 2.0
    return round(max(0.0, min(60.0, max(base, requested_cash))), 2)


def _portfolio_action_plan(action: str, current_weight: float, target_weight: float) -> str:
    delta = target_weight - current_weight
    if abs(delta) < 1.0:
        return "维持"
    if delta > 0:
        return f"增配 {delta:.1f} 个百分点"
    if action == "SELL":
        return f"减配 {abs(delta):.1f} 个百分点"
    return f"回落至目标权重，减配 {abs(delta):.1f} 个百分点"


def build_portfolio_advice(
    *,
    holdings: list[dict],
    user_profile: dict | None = None,
    cash_percent: float = 0.0,
    objective: str = "balanced",
) -> dict:
    """Create portfolio-level suggestions from the latest single-stock advice and learned rules."""
    profile = _normalize_user_profile(user_profile or {})
    holdings_norm = _normalize_portfolio_holdings(holdings)
    if not holdings_norm:
        raise ValueError("组合建议至少需要 1 只持仓或候选股票。")

    risk_max = profile["max_position_per_stock"]
    if profile["risk_profile"] == "conservative":
        effective_max = max(5.0, risk_max * 0.75)
    elif profile["risk_profile"] == "aggressive":
        effective_max = min(100.0, risk_max * 1.15)
    else:
        effective_max = risk_max
    effective_max = round(effective_max, 2)
    target_cash = _target_cash_percent(profile, objective, cash_percent)
    investable_weight = max(0.0, 100.0 - target_cash)
    active_rules = int((load_strategy_rules().get("summary") or {}).get("active_rules", 0) or 0)

    rows = []
    missing_advice = []
    warnings: list[str] = []
    for holding in holdings_norm:
        ticker = holding["ticker"]
        current_weight = holding["weight_percent"]
        latest = get_latest_advice_for_ticker(ticker)
        if not latest:
            missing_advice.append(ticker)
            rows.append(
                {
                    **holding,
                    "action": "HOLD",
                    "confidence": 0.0,
                    "current_weight_percent": current_weight,
                    "target_weight_percent": min(current_weight, effective_max),
                    "delta_percent": round(min(current_weight, effective_max) - current_weight, 2),
                    "plan": "缺少最新个股建议，暂按仓位上限约束处理",
                    "reason": "请先为该标的生成结构化建议。",
                    "data_quality": {},
                    "strategy_rule_adjusted": False,
                }
            )
            continue
        latest = apply_strategy_rules_to_advice(dict(latest))
        rec = latest.get("recommendation", {}) if isinstance(latest.get("recommendation"), dict) else {}
        action = _normalize_action(rec.get("action", "HOLD"))
        confidence = max(0.0, min(1.0, _safe_float(rec.get("confidence"), 0.0)))
        rec_position = max(0.0, min(100.0, _safe_float(rec.get("position_percent"), 0.0)))
        quality = latest.get("data_quality") if isinstance(latest.get("data_quality"), dict) else {}
        quality_score = _safe_float(quality.get("score"), 0.5)

        if action == "BUY":
            target = max(current_weight, rec_position)
            if current_weight <= 0:
                target = min(target, 8.0 if profile["risk_profile"] != "aggressive" else 12.0)
        elif action == "SELL":
            target = 0.0 if confidence >= 0.70 else current_weight * 0.5
        else:
            target = current_weight
            if quality_score < 0.45 and current_weight > effective_max * 0.75:
                target = min(target, effective_max * 0.75)

        if quality_score < 0.45 and action in {"BUY", "SELL"}:
            target *= 0.75
        target = round(max(0.0, min(effective_max, target)), 2)
        rows.append(
            {
                **holding,
                "action": action,
                "confidence": round(confidence, 4),
                "current_weight_percent": current_weight,
                "target_weight_percent": target,
                "delta_percent": round(target - current_weight, 2),
                "plan": _portfolio_action_plan(action, current_weight, target),
                "reason": _clip_text(rec.get("reason"), 260),
                "as_of_date": latest.get("as_of_date", ""),
                "data_quality": {
                    "score": quality.get("score"),
                    "level": quality.get("level"),
                },
                "strategy_rule_adjusted": bool(rec.get("strategy_rule_adjusted")),
            }
        )

    target_sum = sum(_safe_float(row.get("target_weight_percent"), 0.0) for row in rows)
    if target_sum > investable_weight and target_sum > 0:
        scale = investable_weight / target_sum
        for row in rows:
            target = round(_safe_float(row.get("target_weight_percent"), 0.0) * scale, 2)
            row["target_weight_percent"] = target
            row["delta_percent"] = round(target - _safe_float(row.get("current_weight_percent"), 0.0), 2)
            row["plan"] = _portfolio_action_plan(row.get("action", "HOLD"), _safe_float(row.get("current_weight_percent"), 0.0), target)
        warnings.append(f"目标仓位合计超过可投资权重，已按现金底仓 {target_cash:.1f}% 等比例压缩。")

    current_total = sum(_safe_float(row.get("current_weight_percent"), 0.0) for row in rows)
    target_total = sum(_safe_float(row.get("target_weight_percent"), 0.0) for row in rows)
    concentration = [row for row in rows if _safe_float(row.get("current_weight_percent"), 0.0) > effective_max]
    if concentration:
        warnings.append(f"{len(concentration)} 只标的超过当前画像单票上限，建议优先做仓位再平衡。")
    if missing_advice:
        warnings.append(f"{len(missing_advice)} 只标的缺少最新建议：{', '.join(missing_advice[:8])}。")

    rows.sort(key=lambda row: (abs(_safe_float(row.get("delta_percent"), 0.0)), _safe_float(row.get("confidence"), 0.0)), reverse=True)
    return {
        "status": "ok",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "objective": objective,
        "user_profile": profile,
        "cash_percent": round(max(0.0, min(100.0, _safe_float(cash_percent, 0.0))), 2),
        "target_cash_percent": target_cash,
        "current_total_weight_percent": round(current_total, 2),
        "target_total_weight_percent": round(target_total, 2),
        "effective_max_position_per_stock": effective_max,
        "holdings": rows,
        "warnings": warnings,
        "missing_advice": missing_advice,
        "strategy_rule_awareness": {
            "active_rules": active_rules,
            "applied_to_holdings": sum(1 for row in rows if row.get("strategy_rule_adjusted")),
        },
        "limitations": "组合建议基于系统最新个股建议、用户画像与历史错误规则库生成，不构成投资承诺。",
    }


def _action_from_sentiment(value: Any) -> str:
    token = str(value or "").lower().strip()
    if token in {"buy", "long", "positive", "bull", "bullish", "看多", "买入"}:
        return "BUY"
    if token in {"sell", "short", "negative", "bear", "bearish", "看空", "卖出"}:
        return "SELL"
    if "positive" in token or "bull" in token or "看多" in token:
        return "BUY"
    if "negative" in token or "bear" in token or "看空" in token:
        return "SELL"
    return _normalize_action(token)


def _confidence_bucket(value: Any) -> str:
    conf = max(0.0, min(1.0, _safe_float(value, 0.0)))
    lower = min(0.9, int(conf * 10) / 10.0)
    upper = min(1.0, lower + 0.1)
    return f"{lower:.1f}-{upper:.1f}"


def _stock_category(ticker: str) -> str:
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


def _is_directional_action(action: str) -> bool:
    return str(action or "").upper() in {"BUY", "SELL"}


def _directional_hit(action: str, market_move_percent: float) -> bool | None:
    action = _normalize_action(action)
    if action == "BUY":
        return market_move_percent > 0
    if action == "SELL":
        return market_move_percent < 0
    return None


def _directional_return(action: str, market_move_percent: float) -> float | None:
    action = _normalize_action(action)
    if action == "BUY":
        return market_move_percent
    if action == "SELL":
        return -market_move_percent
    return None


def _max_drawdown_percent(returns_percent: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for ret in returns_percent:
        equity *= 1.0 + (_safe_float(ret, 0.0) / 100.0)
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, (equity / peak - 1.0) * 100.0)
    return round(max_drawdown, 4)


def _settlement_key(file_key: str, horizon_days: int) -> str:
    return f"{file_key}::T+{int(horizon_days)}"


def _extract_position_fraction(payload: dict) -> float:
    rec = payload.get("recommendation", {}) or {}
    risk = payload.get("risk", {}) or {}
    raw = rec.get("position_percent")
    if raw is None:
        raw = risk.get("position_percent", 0.0)
    position_pct = max(0.0, min(100.0, _safe_float(raw, 0.0)))
    return position_pct / 100.0


def _fetch_daily_market_df(ticker: str):
    ticker = str(ticker).strip().zfill(6)
    errors: list[str] = []
    try:
        df = ak.stock_zh_a_hist(symbol=ticker, period="daily", adjust="qfq")
        if df is not None and not df.empty:
            return df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "volume",
                    "成交额": "amount",
                    "换手率": "turnover",
                }
            ), "akshare_eastmoney_hist", errors
        errors.append("akshare_eastmoney_hist returned empty data")
    except Exception as exc:
        errors.append(f"akshare_eastmoney_hist: {exc}")

    prefix = "sh" if ticker.startswith("6") else "sz"
    symbol = f"{prefix}{ticker}"
    try:
        df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
        if df is not None and not df.empty:
            return df, "akshare_sina_daily", errors
        errors.append("akshare_sina_daily returned empty data")
    except Exception as exc:
        errors.append(f"akshare_sina_daily: {exc}")
    return None, "", errors


def _load_price_map_for_ticker(ticker: str) -> dict[str, float]:
    df, _, _ = _fetch_daily_market_df(ticker)
    if df is None or df.empty:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        date = str(row.get("date", "")).split(" ")[0]
        close = _safe_float(row.get("close"), default=float("nan"))
        if date and close == close:
            out[date] = close
    return out


def get_market_ohlc_bars(ticker: str, limit: int = 40) -> dict:
    """Return recent daily OHLC bars for mini K-line chart in the UI."""
    ticker = str(ticker).strip().zfill(6)
    limit = max(2, min(120, int(limit)))
    df, source, errors = _fetch_daily_market_df(ticker)
    if df is None or df.empty:
        return {"ticker": ticker, "bars": [], "error": "; ".join(errors) or "no market data"}

    bars: list[dict] = []
    for _, row in df.tail(limit).iterrows():
        date = str(row.get("date", "")).split(" ")[0]
        open_px = _safe_float(row.get("open"), default=float("nan"))
        high_px = _safe_float(row.get("high"), default=float("nan"))
        low_px = _safe_float(row.get("low"), default=float("nan"))
        close_px = _safe_float(row.get("close"), default=float("nan"))
        if not date or not all(x == x for x in (open_px, high_px, low_px, close_px)):
            continue
        bars.append(
            {
                "date": date,
                "open": round(open_px, 4),
                "high": round(high_px, 4),
                "low": round(low_px, 4),
                "close": round(close_px, 4),
            }
        )
    return {"ticker": ticker, "bars": bars, "count": len(bars), "source": source}


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


def _trade_date_after(dates_sorted: list[str], as_of_date: str, horizon_days: int) -> str:
    future_dates = [d for d in dates_sorted if d > as_of_date]
    if len(future_dates) < horizon_days:
        return ""
    return future_dates[horizon_days - 1]


def _close_series_until(price_map: dict[str, float], end_date: str, limit: int = 80) -> list[tuple[str, float]]:
    rows = []
    for date in sorted(price_map.keys()):
        if date > end_date:
            break
        close = _safe_float(price_map.get(date), default=float("nan"))
        if close == close and close > 0:
            rows.append((date, close))
    return rows[-limit:]


def _baseline_action_returns(
    *,
    price_map: dict[str, float],
    as_of_date: str,
    settled_date: str,
    market_move_percent: float,
) -> dict:
    history = _close_series_until(price_map, as_of_date, limit=80)
    close_t = _safe_float(price_map.get(as_of_date), default=float("nan"))
    close_target = _safe_float(price_map.get(settled_date), default=float("nan"))
    buy_hold_return = market_move_percent
    hold_return = 0.0

    closes = [x[1] for x in history]
    ma20_action = "HOLD"
    ma20_return = 0.0
    if len(closes) >= 20:
        ma20 = sum(closes[-20:]) / 20.0
        ma20_action = "BUY" if close_t > ma20 else "HOLD"
        ma20_return = buy_hold_return if ma20_action == "BUY" else 0.0

    rsi_action = "HOLD"
    rsi_return = 0.0
    if len(closes) >= 15:
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        recent = deltas[-14:]
        gains = [max(x, 0.0) for x in recent]
        losses = [max(-x, 0.0) for x in recent]
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        if avg_loss == 0 and avg_gain > 0:
            rsi = 100.0
        elif avg_loss == 0:
            rsi = 50.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)
        if rsi <= 35:
            rsi_action = "BUY"
            rsi_return = buy_hold_return
        elif rsi >= 70:
            rsi_action = "SELL"
            rsi_return = -buy_hold_return

    macd_action = "HOLD"
    macd_return = 0.0
    if len(closes) >= 26:
        def _ema(values: list[float], span: int) -> float:
            alpha = 2.0 / (span + 1.0)
            ema = values[0]
            for value in values[1:]:
                ema = alpha * value + (1.0 - alpha) * ema
            return ema

        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        if ema12 > ema26:
            macd_action = "BUY"
            macd_return = buy_hold_return
        elif ema12 < ema26:
            macd_action = "SELL"
            macd_return = -buy_hold_return

    baselines = {
        "buy_hold": {"action": "BUY", "return_percent": round(buy_hold_return, 4)},
        "always_hold": {"action": "HOLD", "return_percent": round(hold_return, 4)},
        "ma20": {"action": ma20_action, "return_percent": round(ma20_return, 4)},
        "rsi": {"action": rsi_action, "return_percent": round(rsi_return, 4)},
        "macd": {"action": macd_action, "return_percent": round(macd_return, 4)},
    }
    directional = [
        value["return_percent"]
        for key, value in baselines.items()
        if key != "always_hold" and _is_directional_action(str(value.get("action")))
    ]
    simple_avg = sum(directional) / len(directional) if directional else 0.0
    baselines["simple_directional_avg"] = {"action": "MIXED", "return_percent": round(simple_avg, 4)}
    best_name, best_payload = max(baselines.items(), key=lambda item: _safe_float(item[1].get("return_percent"), 0.0))
    baselines["_summary"] = {
        "best_baseline": best_name,
        "best_baseline_return": round(_safe_float(best_payload.get("return_percent"), 0.0), 4),
        "valid_history_points": len(history),
        "close_t": round(close_t, 4) if close_t == close_t else None,
        "close_target": round(close_target, 4) if close_target == close_target else None,
    }
    return baselines


def _infer_mistake_attribution(record: dict, payload: dict, baselines: dict) -> dict:
    action = _normalize_action(record.get("decision", "HOLD"))
    pnl = _safe_float(record.get("pnl_percent"), 0.0)
    market_move = _safe_float(record.get("market_move_percent"), 0.0)
    confidence = _safe_float(record.get("confidence"), 0.0)
    rec = payload.get("recommendation", {}) or {}
    quality = payload.get("data_quality", {}) or {}
    technical = (payload.get("analyst_cases", {}) or {}).get("technical_flow", {}) or {}
    fundamental = (payload.get("analyst_cases", {}) or {}).get("fundamental_news", {}) or {}
    tech_action = _action_from_sentiment(technical.get("sentiment") or technical.get("action"))
    fund_action = _action_from_sentiment(fundamental.get("sentiment") or fundamental.get("action"))
    data_score = _safe_float(quality.get("score"), 0.5)
    baseline_summary = baselines.get("_summary", {}) if isinstance(baselines, dict) else {}
    best_baseline_return = _safe_float(baseline_summary.get("best_baseline_return"), 0.0)
    agent_return = _safe_float(record.get("pnl_percent"), 0.0)
    tags: list[str] = []

    if action == "HOLD" and abs(market_move) >= 2.0:
        mistake_type = "missed_opportunity" if market_move > 0 else "avoided_loss"
    elif pnl >= 0:
        mistake_type = "no_mistake"
    elif data_score < 0.45:
        mistake_type = "low_quality_overreach"
        tags.append("数据质量不足")
    elif action == "BUY" and market_move < 0:
        mistake_type = "false_breakout"
        tags.append("买入后价格反向")
    elif action == "SELL" and market_move > 0:
        mistake_type = "over_defensive_sell"
        tags.append("卖出后价格反弹")
    else:
        mistake_type = "directional_miss"

    if tech_action in {"BUY", "SELL"} and fund_action in {"BUY", "SELL"} and tech_action != fund_action:
        tags.append("技术面与基本面冲突")
    if confidence >= 0.75 and pnl < 0:
        tags.append("高置信误判")
    if best_baseline_return > agent_return + 0.5:
        tags.append("跑输简单基准")

    if mistake_type == "no_mistake":
        root_cause = "建议兑现为正收益，未识别到明显错误。"
        future_rule = "保留当前证据组合，但继续监控置信度校准。"
    elif mistake_type == "missed_opportunity":
        root_cause = "系统保持观望，但后续价格明显上涨，可能低估了趋势延续或催化强度。"
        future_rule = "若技术特征和数据质量同时较强，可允许小仓位试探而非完全观望。"
    elif mistake_type == "avoided_loss":
        root_cause = "系统观望期间价格下跌，当前保守处理避免了方向性亏损。"
        future_rule = "保留观望策略，并记录为风险规避成功样本。"
    elif mistake_type == "low_quality_overreach":
        root_cause = "数据质量偏低时仍给出方向性建议，证据不足导致兑现亏损。"
        future_rule = "数据质量低于0.45时，方向性仓位应继续压低或转为HOLD。"
    elif mistake_type == "false_breakout":
        root_cause = "买入后价格未延续，可能是假突破或量价持续性不足。"
        future_rule = "突破类买入需增加次日确认、量能延续或回踩不破条件。"
    elif mistake_type == "over_defensive_sell":
        root_cause = "卖出后价格反弹，可能过度放大了短线风险或忽略趋势修复。"
        future_rule = "SELL 需结合趋势破位确认，未破关键均线时优先减仓而非强卖。"
    else:
        root_cause = "建议方向与后续价格方向不一致，需复查主导证据和风控约束。"
        future_rule = "提高冲突场景下的仓位折扣，并检查主导Agent的历史可靠性。"

    return {
        "mistake_type": mistake_type,
        "severity": "high" if pnl < -2.0 or (confidence >= 0.75 and pnl < 0) else ("medium" if pnl < 0 or mistake_type == "missed_opportunity" else "low"),
        "root_cause": root_cause,
        "future_rule": future_rule,
        "tags": tags,
        "agent_missed": [
            name
            for name, signal in {
                "technical_flow": {"action": tech_action},
                "fundamental_news": {"action": fund_action},
                "recommendation": {"action": action},
            }.items()
            if _is_directional_action(signal["action"]) and _directional_hit(signal["action"], market_move) is False
        ],
        "baseline_gap_percent": round(agent_return - best_baseline_return, 4),
        "recommendation_reason": _clip_text(rec.get("reason"), 180),
    }


def _extract_signal_profile(payload: dict, ticker: str) -> dict:
    rec = payload.get("recommendation", {}) or {}
    referee = payload.get("referee", {}) or {}
    analysts = payload.get("analyst_cases", {}) or {}
    agents: dict[str, dict] = {}

    for name, case in analysts.items():
        if not isinstance(case, dict):
            continue
        sentiment = case.get("sentiment") or case.get("decision") or case.get("action")
        action = _action_from_sentiment(sentiment)
        agents[str(name)] = {
            "agent": str(name),
            "sentiment": str(sentiment or ""),
            "action": action,
            "confidence": round(max(0.0, min(1.0, _safe_float(case.get("confidence"), 0.0))), 4),
        }

    technical = agents.get("technical_flow", {})
    fundamental = agents.get("fundamental_news", {})
    technical_action = str(technical.get("action", "HOLD"))
    fundamental_action = str(fundamental.get("action", "HOLD"))
    conflict = (
        _is_directional_action(technical_action)
        and _is_directional_action(fundamental_action)
        and technical_action != fundamental_action
    )

    return {
        "recommendation": {
            "action": _normalize_action(rec.get("action", "HOLD")),
            "confidence": round(max(0.0, min(1.0, _safe_float(rec.get("confidence"), 0.0))), 4),
        },
        "referee": {
            "action": _normalize_action(referee.get("decision") or referee.get("action") or rec.get("action", "HOLD")),
            "confidence": round(max(0.0, min(1.0, _safe_float(referee.get("confidence"), 0.0))), 4),
        },
        "agents": agents,
        "conflict": {
            "technical_vs_fundamental": conflict,
            "technical_action": technical_action,
            "fundamental_action": fundamental_action,
        },
        "stock_category": _stock_category(ticker),
    }


def _metric_summary(rows: list[dict], *, horizon: str = "", confidence_bucket: str = "") -> dict:
    ordered = sorted(rows, key=lambda r: (str(r.get("as_of_date", "")), str(r.get("settled_date", "")), str(r.get("ticker", ""))))
    returns = [_safe_float(r.get("pnl_percent"), 0.0) for r in ordered]
    excess_buy_hold = [_safe_float(r.get("excess_return_vs_buy_hold"), 0.0) for r in ordered if r.get("excess_return_vs_buy_hold") is not None]
    excess_best = [_safe_float(r.get("excess_return_vs_best_baseline"), 0.0) for r in ordered if r.get("excess_return_vs_best_baseline") is not None]
    directional_rows = [r for r in ordered if _is_directional_action(str(r.get("decision", "")))]
    wins = sum(1 for r in directional_rows if _safe_float(r.get("pnl_percent"), 0.0) > 0)
    out = {
        "samples": len(ordered),
        "directional_samples": len(directional_rows),
        "hit_rate": round(wins / len(directional_rows), 4) if directional_rows else None,
        "avg_return": round(sum(returns) / len(returns), 4) if returns else None,
        "avg_excess_vs_buy_hold": round(sum(excess_buy_hold) / len(excess_buy_hold), 4) if excess_buy_hold else None,
        "avg_excess_vs_best_baseline": round(sum(excess_best) / len(excess_best), 4) if excess_best else None,
        "max_drawdown": _max_drawdown_percent(returns) if returns else None,
    }
    if horizon:
        out["horizon"] = horizon
    if confidence_bucket:
        out["confidence_bucket"] = confidence_bucket
    return out


def build_advice_settlement_evaluation(records: list[dict]) -> dict:
    rows = []
    for record in records:
        if not isinstance(record, dict) or str(record.get("source", "")).strip() != "advice_settlement":
            continue
        row = dict(record)
        if not str(row.get("horizon", "")).strip():
            row["horizon"] = "T+1"
            row["horizon_days"] = 1
        if not str(row.get("confidence_bucket", "")).strip():
            row["confidence_bucket"] = _confidence_bucket(row.get("confidence", 0.0))
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("horizon", "")), str(r.get("as_of_date", "")), str(r.get("ticker", ""))))
    if not rows:
        return {
            "horizons": [],
            "confidence_calibration": [],
            "agent_reliability": [],
            "conflict_metrics": [],
            "stock_type_metrics": [],
            "samples": 0,
        }

    by_horizon: dict[str, list[dict]] = defaultdict(list)
    by_horizon_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_horizon_stock: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        horizon = str(row.get("horizon", ""))
        bucket = str(row.get("confidence_bucket") or _confidence_bucket(row.get("confidence", 0.0)))
        stock_category = str((row.get("signal_profile") or {}).get("stock_category") or _stock_category(str(row.get("ticker", ""))))
        by_horizon[horizon].append(row)
        by_horizon_bucket[(horizon, bucket)].append(row)
        by_horizon_stock[(horizon, stock_category)].append(row)

    horizons: list[dict] = []
    for horizon in sorted(by_horizon.keys(), key=lambda x: int(x.replace("T+", "") or 0)):
        horizon_rows = by_horizon[horizon]
        summary = _metric_summary(horizon_rows, horizon=horizon)
        bucket_metrics = []
        for (bucket_horizon, bucket), bucket_rows in sorted(by_horizon_bucket.items()):
            if bucket_horizon != horizon:
                continue
            metric = _metric_summary(bucket_rows, horizon=horizon, confidence_bucket=bucket)
            bucket_metrics.append(metric)
        summary["confidence_buckets"] = bucket_metrics
        horizons.append(summary)

    confidence_calibration: list[dict] = []
    for (horizon, bucket), bucket_rows in sorted(by_horizon_bucket.items(), key=lambda x: (int(x[0][0].replace("T+", "") or 0), x[0][1])):
        directional_rows = [r for r in bucket_rows if _is_directional_action(str(r.get("decision", "")))]
        if not directional_rows:
            continue
        hit_rate = sum(1 for r in directional_rows if _safe_float(r.get("pnl_percent"), 0.0) > 0) / len(directional_rows)
        avg_confidence = sum(_safe_float(r.get("confidence"), 0.0) for r in directional_rows) / len(directional_rows)
        confidence_calibration.append(
            {
                "horizon": horizon,
                "confidence_bucket": bucket,
                "samples": len(directional_rows),
                "avg_confidence": round(avg_confidence, 4),
                "hit_rate": round(hit_rate, 4),
                "calibration_error": round(hit_rate - avg_confidence, 4),
                "abs_calibration_error": round(abs(hit_rate - avg_confidence), 4),
                "avg_return": round(
                    sum(_safe_float(r.get("pnl_percent"), 0.0) for r in directional_rows) / len(directional_rows),
                    4,
                ),
            }
        )

    agent_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    conflict_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        horizon = str(row.get("horizon", ""))
        market_move = _safe_float(row.get("market_move_percent"), 0.0)
        profile = row.get("signal_profile") if isinstance(row.get("signal_profile"), dict) else {}
        agents = profile.get("agents", {}) if isinstance(profile.get("agents"), dict) else {}
        for agent, signal in agents.items():
            if not isinstance(signal, dict):
                continue
            action = _normalize_action(signal.get("action", "HOLD"))
            signal_return = _directional_return(action, market_move)
            if signal_return is None:
                continue
            agent_rows[(horizon, str(agent))].append(
                {
                    "hit": bool(_directional_hit(action, market_move)),
                    "return": signal_return,
                    "confidence": _safe_float(signal.get("confidence"), 0.0),
                }
            )

        conflict = profile.get("conflict", {}) if isinstance(profile.get("conflict"), dict) else {}
        if conflict.get("technical_vs_fundamental"):
            technical_action = _normalize_action(conflict.get("technical_action", "HOLD"))
            fundamental_action = _normalize_action(conflict.get("fundamental_action", "HOLD"))
            conflict_rows[horizon].append(
                {
                    "technical_hit": bool(_directional_hit(technical_action, market_move)),
                    "fundamental_hit": bool(_directional_hit(fundamental_action, market_move)),
                    "market_move_percent": market_move,
                }
            )

    agent_reliability = []
    for (horizon, agent), items in sorted(agent_rows.items(), key=lambda x: (int(x[0][0].replace("T+", "") or 0), x[0][1])):
        agent_reliability.append(
            {
                "horizon": horizon,
                "agent": agent,
                "samples": len(items),
                "hit_rate": round(sum(1 for x in items if x["hit"]) / len(items), 4),
                "avg_return": round(sum(_safe_float(x.get("return"), 0.0) for x in items) / len(items), 4),
                "avg_confidence": round(sum(_safe_float(x.get("confidence"), 0.0) for x in items) / len(items), 4),
            }
        )

    conflict_metrics = []
    for horizon, items in sorted(conflict_rows.items(), key=lambda x: int(x[0].replace("T+", "") or 0)):
        if not items:
            continue
        tech_hits = sum(1 for x in items if x.get("technical_hit"))
        fund_hits = sum(1 for x in items if x.get("fundamental_hit"))
        conflict_metrics.append(
            {
                "horizon": horizon,
                "samples": len(items),
                "technical_hit_rate": round(tech_hits / len(items), 4),
                "fundamental_hit_rate": round(fund_hits / len(items), 4),
                "better_side": "technical" if tech_hits > fund_hits else ("fundamental" if fund_hits > tech_hits else "tie"),
            }
        )

    stock_type_metrics = []
    for (horizon, category), category_rows in sorted(by_horizon_stock.items(), key=lambda x: (int(x[0][0].replace("T+", "") or 0), x[0][1])):
        metric = _metric_summary(category_rows, horizon=horizon)
        metric["stock_category"] = category
        stock_type_metrics.append(metric)

    return {
        "horizons": horizons,
        "confidence_calibration": confidence_calibration,
        "agent_reliability": agent_reliability,
        "conflict_metrics": conflict_metrics,
        "stock_type_metrics": stock_type_metrics,
        "samples": len(rows),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _quality_score_from_rows(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    directional = [r for r in rows if _is_directional_action(str(r.get("decision", "")))]
    hit_rate = (
        sum(1 for r in directional if _safe_float(r.get("pnl_percent"), 0.0) > 0) / len(directional)
        if directional
        else 0.5
    )
    avg_return = sum(_safe_float(r.get("pnl_percent"), 0.0) for r in rows) / len(rows)
    avg_excess = sum(_safe_float(r.get("excess_return_vs_best_baseline"), 0.0) for r in rows) / len(rows)
    avg_reward = sum(_safe_float(r.get("reward_score"), 0.0) for r in rows) / len(rows)
    max_dd = _max_drawdown_percent([_safe_float(r.get("pnl_percent"), 0.0) for r in rows])
    overconfident_losses = sum(
        1
        for r in rows
        if _safe_float(r.get("confidence"), 0.0) >= 0.75 and _safe_float(r.get("pnl_percent"), 0.0) < 0
    )
    score = 50.0
    score += (hit_rate - 0.5) * 35.0
    score += max(-12.0, min(12.0, avg_return * 2.0))
    score += max(-12.0, min(12.0, avg_excess * 1.8))
    score += max(-8.0, min(8.0, avg_reward * 6.0))
    score += max(-10.0, min(0.0, max_dd * 0.6))
    score -= min(12.0, overconfident_losses * 2.5)
    return round(max(0.0, min(100.0, score)), 2)


def _analytics_mistake_attribution(record: dict) -> dict:
    existing = record.get("mistake_attribution") if isinstance(record.get("mistake_attribution"), dict) else {}
    existing_type = str(existing.get("mistake_type", "") or "").strip()
    if existing_type:
        return existing

    action = _normalize_action(record.get("decision", "HOLD"))
    pnl = _safe_float(record.get("pnl_percent"), 0.0)
    market_move = _safe_float(record.get("market_move_percent"), 0.0)
    excess_best = _safe_float(record.get("excess_return_vs_best_baseline"), 0.0)
    confidence = _safe_float(record.get("confidence"), 0.0)

    if action == "HOLD" and abs(market_move) >= 2.0:
        mistake_type = "missed_opportunity" if market_move > 0 else "avoided_loss"
    elif pnl >= 0:
        mistake_type = "no_mistake"
    elif action == "BUY" and (market_move < 0 or excess_best < -0.5):
        mistake_type = "false_breakout"
    elif action == "SELL" and market_move > 0:
        mistake_type = "over_defensive_sell"
    else:
        mistake_type = "directional_miss"

    root_cause_map = {
        "no_mistake": "历史样本缺少显式归因；按兑现收益为正自动归为未识别到明显错误。",
        "false_breakout": "历史样本缺少显式归因；按买入后收益/相对基准转弱自动归为突破持续性不足。",
        "over_defensive_sell": "历史样本缺少显式归因；按卖出后市场上行自动归为防守过度。",
        "missed_opportunity": "历史样本缺少显式归因；按观望后市场明显上涨自动归为错失机会。",
        "avoided_loss": "历史样本缺少显式归因；按观望后市场明显下跌自动归为规避亏损。",
        "directional_miss": "历史样本缺少显式归因；按建议方向与兑现表现不一致自动归为方向性偏差。",
    }
    return {
        "mistake_type": mistake_type,
        "severity": "high" if pnl < -2.0 or (confidence >= 0.75 and pnl < 0) else ("medium" if pnl < 0 else "low"),
        "root_cause": root_cause_map.get(mistake_type, "历史样本自动补归因。"),
        "future_rule": "",
        "tags": ["历史补归因"],
        "inferred_for_analytics": True,
    }


def _load_settlement_records_for_analytics() -> list[dict]:
    state = _load_settlement_state()
    records = state.get("records", []) if isinstance(state.get("records"), list) else []
    if records:
        return [r for r in records if isinstance(r, dict)]
    reflections = read_json(REFLECTIONS_PATH, [])
    return [r for r in reflections if isinstance(r, dict) and str(r.get("source", "")) == "advice_settlement"] if isinstance(reflections, list) else []


def build_advice_quality_ranking(limit: int = 20, horizon: str | None = None, min_samples: int = 1) -> dict:
    limit = max(1, min(200, int(limit)))
    min_samples = max(1, int(min_samples))
    horizon_filter = str(horizon or "").strip().upper()
    rows = []
    for record in _load_settlement_records_for_analytics():
        if str(record.get("source", "")).strip() != "advice_settlement":
            continue
        if horizon_filter and str(record.get("horizon", "")).upper() != horizon_filter:
            continue
        rows.append(record)

    by_ticker: dict[str, list[dict]] = defaultdict(list)
    by_horizon: dict[str, list[dict]] = defaultdict(list)
    mistake_counter: dict[str, int] = defaultdict(int)
    inferred_attribution_count = 0
    for row in rows:
        ticker = str(row.get("ticker", "")).zfill(6)
        if ticker:
            by_ticker[ticker].append(row)
        by_horizon[str(row.get("horizon", "T+1"))].append(row)
        attribution = _analytics_mistake_attribution(row)
        if attribution.get("inferred_for_analytics"):
            inferred_attribution_count += 1
        mistake_type = str(attribution.get("mistake_type", "") or "")
        if mistake_type:
            mistake_counter[mistake_type] += 1

    ranking = []
    for ticker, ticker_rows in by_ticker.items():
        if len(ticker_rows) < min_samples:
            continue
        metric = _metric_summary(ticker_rows)
        score = _quality_score_from_rows(ticker_rows)
        mistake_types: dict[str, int] = defaultdict(int)
        error_mistake_types: dict[str, int] = defaultdict(int)
        inferred_for_ticker = 0
        for row in ticker_rows:
            attribution = _analytics_mistake_attribution(row)
            if attribution.get("inferred_for_analytics"):
                inferred_for_ticker += 1
            mistake_type = str(attribution.get("mistake_type", "") or "unattributed")
            mistake_types[mistake_type] += 1
            if mistake_type not in {"no_mistake", "avoided_loss"}:
                error_mistake_types[mistake_type] += 1
        latest = max(ticker_rows, key=lambda r: str(r.get("settled_at", "")))
        latest_attribution = _analytics_mistake_attribution(latest)
        dominant_pool = error_mistake_types or mistake_types
        ranking.append(
            {
                "ticker": ticker,
                "quality_score": score,
                "samples": len(ticker_rows),
                "directional_samples": metric.get("directional_samples"),
                "hit_rate": metric.get("hit_rate"),
                "avg_return": metric.get("avg_return"),
                "avg_excess_vs_buy_hold": metric.get("avg_excess_vs_buy_hold"),
                "avg_excess_vs_best_baseline": metric.get("avg_excess_vs_best_baseline"),
                "max_drawdown": metric.get("max_drawdown"),
                "latest_horizon": latest.get("horizon", ""),
                "latest_mistake_type": latest_attribution.get("mistake_type", ""),
                "dominant_mistake_type": max(dominant_pool.items(), key=lambda x: x[1])[0] if dominant_pool else "",
                "inferred_attribution_count": inferred_for_ticker,
            }
        )
    ranking.sort(key=lambda x: (x["quality_score"], x["samples"]), reverse=True)

    baseline_by_horizon = []
    for horizon_name, horizon_rows in sorted(by_horizon.items(), key=lambda x: int(str(x[0]).replace("T+", "") or 0)):
        metric = _metric_summary(horizon_rows, horizon=horizon_name)
        baseline_by_horizon.append(metric)

    worst_cases = sorted(
        [
            {
                "ticker": str(row.get("ticker", "")).zfill(6),
                "horizon": row.get("horizon", ""),
                "pnl_percent": row.get("pnl_percent"),
                "excess_return_vs_best_baseline": row.get("excess_return_vs_best_baseline"),
                "decision": row.get("decision"),
                "confidence": row.get("confidence"),
                "mistake_attribution": _analytics_mistake_attribution(row),
                "as_of_date": row.get("as_of_date", ""),
                "settled_date": row.get("settled_date", ""),
            }
            for row in rows
            if _analytics_mistake_attribution(row).get("mistake_type") not in {"", "no_mistake", "avoided_loss"}
        ],
        key=lambda x: _safe_float(x.get("pnl_percent"), 0.0),
    )[:limit]

    return {
        "ranking": ranking[:limit],
        "baseline_comparison": baseline_by_horizon,
        "mistake_distribution": dict(sorted(mistake_counter.items(), key=lambda x: x[1], reverse=True)),
        "worst_cases": worst_cases,
        "samples": len(rows),
        "inferred_attribution_count": inferred_attribution_count,
        "horizon": horizon_filter,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def simulate_advice_counterfactual(
    *,
    ticker: str,
    scenario: dict,
    user_profile: dict | None = None,
) -> dict:
    ticker = str(ticker).strip().zfill(6)
    advice = get_latest_advice_for_ticker(ticker)
    if not advice:
        raise FileNotFoundError(f"未找到 {ticker} 的最新建议，请先生成建议。")
    scenario = scenario if isinstance(scenario, dict) else {}
    rec = advice.get("recommendation", {}) or {}
    risk = advice.get("risk", {}) or {}
    original_action = _normalize_action(rec.get("action", "HOLD"))
    original_position = max(0.0, min(100.0, _safe_float(rec.get("position_percent"), risk.get("position_percent", 0.0))))
    original_confidence = max(0.0, min(1.0, _safe_float(rec.get("confidence"), 0.5)))
    profile = _normalize_user_profile(user_profile or advice.get("user_profile") or {})

    price_change = _safe_float(scenario.get("price_change_pct"), 0.0)
    market_index_change = _safe_float(scenario.get("market_index_change_pct"), 0.0)
    user_position = max(0.0, min(100.0, _safe_float(scenario.get("user_position_percent"), profile.get("current_position_percent", 0.0))))
    volume_change = str(scenario.get("volume_change", "") or "").strip().lower()
    news_sentiment = str(scenario.get("news_sentiment", "") or "").strip().lower()
    breaks_ma20 = bool(scenario.get("breaks_ma20", False))
    breaks_stop_loss = bool(scenario.get("breaks_stop_loss", False))

    score = 0.0
    reasons: list[str] = []
    if price_change >= 3:
        score += 0.18
        reasons.append("情景价格明显上涨，短线动量增强。")
    elif price_change <= -3:
        score -= 0.22
        reasons.append("情景价格明显下跌，短线结构承压。")
    if market_index_change <= -1.5:
        score -= 0.14
        reasons.append("大盘同步走弱，系统性风险上升。")
    elif market_index_change >= 1.5:
        score += 0.08
        reasons.append("大盘走强，对个股风险偏好有支撑。")
    if "放量" in volume_change or "high" in volume_change or "increase" in volume_change:
        score += 0.08 if price_change >= 0 else -0.08
        reasons.append("放量会放大当前价格方向的可信度。")
    if "缩量" in volume_change or "low" in volume_change or "decrease" in volume_change:
        score -= 0.05 if price_change >= 0 else 0.0
        reasons.append("缩量上涨的持续性需要打折。")
    if news_sentiment in {"negative", "bearish", "看空", "利空"}:
        score -= 0.16
        reasons.append("新闻/事件面转负，降低方向性仓位。")
    elif news_sentiment in {"positive", "bullish", "看多", "利好"}:
        score += 0.12
        reasons.append("新闻/事件面转正，提高买入条件。")
    if breaks_ma20 or breaks_stop_loss:
        score -= 0.24
        reasons.append("跌破关键均线或止损条件，风控优先。")
    if profile["risk_profile"] == "conservative":
        score -= 0.06
    elif profile["risk_profile"] == "aggressive":
        score += 0.05

    base_score = {"BUY": 0.18, "HOLD": 0.0, "SELL": -0.18}.get(original_action, 0.0)
    final_score = base_score + score
    simulated_action = "BUY" if final_score >= 0.16 else ("SELL" if final_score <= -0.16 else "HOLD")
    if breaks_stop_loss:
        simulated_action = "SELL" if user_position > 0 else "HOLD"
    simulated_position = original_position
    if simulated_action == "BUY":
        simulated_position = max(original_position, 10.0)
        if final_score > 0.35:
            simulated_position = max(simulated_position, 25.0)
        simulated_position = min(simulated_position, profile["max_position_per_stock"])
    elif simulated_action == "SELL":
        simulated_position = 0.0 if breaks_stop_loss else max(0.0, min(user_position, original_position) * 0.5)
    else:
        simulated_position = min(user_position if user_position > 0 else original_position, profile["max_position_per_stock"])
    simulated_confidence = max(0.2, min(0.95, original_confidence + abs(score) * 0.45 - (0.08 if not reasons else 0.0)))

    if not reasons:
        reasons.append("情景输入未显著改变原始证据结构，建议保持原计划。")
    changed = simulated_action != original_action or abs(simulated_position - original_position) >= 5.0
    return {
        "ticker": ticker,
        "original_action": original_action,
        "original_position_percent": round(original_position, 2),
        "original_confidence": round(original_confidence, 4),
        "simulated_action": simulated_action,
        "simulated_position_percent": round(simulated_position, 2),
        "simulated_confidence": round(simulated_confidence, 4),
        "changed": changed,
        "reason": " ".join(reasons),
        "scenario_score_delta": round(score, 4),
        "scenario": {
            "price_change_pct": price_change,
            "market_index_change_pct": market_index_change,
            "volume_change": scenario.get("volume_change", ""),
            "news_sentiment": scenario.get("news_sentiment", ""),
            "user_position_percent": user_position,
            "breaks_ma20": breaks_ma20,
            "breaks_stop_loss": breaks_stop_loss,
        },
        "risk_note": "反事实推演只用于交易计划讨论，真实行情需重新生成建议确认。",
    }


def settle_advice_experience(base_dir: str, max_items: int = 2000, horizons: list[int] | None = None) -> dict:
    _ = base_dir  # 与其他 service 保持签名一致，当前逻辑使用全局路径即可
    selected_horizons = []
    for raw_horizon in horizons or list(ADVICE_EVALUATION_HORIZONS):
        try:
            horizon_days = int(raw_horizon)
        except Exception:
            continue
        if horizon_days in ADVICE_EVALUATION_HORIZONS:
            selected_horizons.append(horizon_days)
    selected_horizons = list(dict.fromkeys(selected_horizons))
    if not selected_horizons:
        selected_horizons = list(ADVICE_EVALUATION_HORIZONS)

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
        ticker = str(payload.get("ticker", "")).zfill(6)
        action = _normalize_action((payload.get("recommendation", {}) or {}).get("action", "HOLD"))
        confidence = max(0.0, min(1.0, _safe_float((payload.get("recommendation", {}) or {}).get("confidence"), 0.0)))
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

        position = _extract_position_fraction(payload)
        signal_profile = _extract_signal_profile(payload, ticker)
        for horizon_days in selected_horizons:
            horizon = f"T+{horizon_days}"
            horizon_key = _settlement_key(file_key, horizon_days)
            # Backward compatibility: historical state used the bare file name
            # for T+1 records. Treat it as settled only for that horizon.
            if horizon_key in settled_keys or (horizon_days == 1 and file_key in settled_keys):
                continue

            settled_date = _trade_date_after(dates_sorted, as_of_date, horizon_days)
            if not settled_date:
                pending_count += 1
                continue

            close_target = _safe_float(price_map.get(settled_date), default=float("nan"))
            if close_target != close_target or close_target <= 0:
                pending_count += 1
                continue

            market_move_percent = (close_target - close_t) / close_t * 100.0
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
                "confidence": round(confidence, 4),
                "confidence_bucket": _confidence_bucket(confidence),
                "horizon": horizon,
                "horizon_days": horizon_days,
                "pnl_percent": round(actual_pnl_percent, 4),
                "market_move_percent": round(market_move_percent, 4),
                "directional_hit": _directional_hit(action, market_move_percent),
                "position": round(position, 4),
                "reward_score": round(reward.reward, 4),
                "reward_text": (
                    f"reward={reward.reward:.3f}, return={reward.return_component:.3f}, "
                    f"risk_penalty={reward.volatility_penalty + reward.drawdown_penalty + reward.loss_streak_penalty:.3f}, "
                    f"behavior_penalty={reward.turnover_penalty + reward.exposure_penalty + reward.hold_penalty + reward.hold_streak_penalty:.3f}"
                ),
                "reward_stats": reward.to_dict(),
                "reflection_text": (
                    f"建议兑现: {as_of_date} -> {settled_date}，周期={horizon}，动作={action}，"
                    f"收益={actual_pnl_percent:.2f}%。"
                ),
                "math_stats": f"{horizon}收益结算，标的波动={market_move_percent:.2f}%，仓位={position*100:.1f}%。",
                "source": "advice_settlement",
                "advice_file": file_key,
                "settlement_key": horizon_key,
                "signal_profile": signal_profile,
                "as_of_date": as_of_date,
                "settled_date": settled_date,
                "settled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            baseline_returns = _baseline_action_returns(
                price_map=price_map,
                as_of_date=as_of_date,
                settled_date=settled_date,
                market_move_percent=market_move_percent,
            )
            record["baseline_returns"] = baseline_returns
            record["excess_return_vs_buy_hold"] = round(actual_pnl_percent - _safe_float(baseline_returns.get("buy_hold", {}).get("return_percent"), 0.0), 4)
            record["excess_return_vs_best_baseline"] = round(
                actual_pnl_percent - _safe_float((baseline_returns.get("_summary", {}) or {}).get("best_baseline_return"), 0.0),
                4,
            )
            record["mistake_attribution"] = _infer_mistake_attribution(record, payload, baseline_returns)
            reflections.append(record)
            new_records.append(record)
            ticker_hist_pnls[ticker].append(actual_pnl_percent)
            ticker_recent_actions[ticker].append(action)
            settled_keys.add(horizon_key)
            settled_count += 1

    existing_rule_library = load_strategy_rules()
    existing_rule_summary = existing_rule_library.get("summary") if isinstance(existing_rule_library.get("summary"), dict) else {}
    strategy_rule_update = {
        "inserted": 0,
        "updated": 0,
        "total_rules": len(existing_rule_library.get("rules", [])),
        "active_rules": int(existing_rule_summary.get("active_rules", 0) or 0),
    }
    if new_records:
        write_json(REFLECTIONS_PATH, reflections)
        strategy_rule_update = update_strategy_rules_from_records(new_records)
    state["settled_keys"] = list(settled_keys)
    state["records"] = (state.get("records") or []) + new_records
    state["evaluation"] = build_advice_settlement_evaluation(state["records"])
    state["strategy_rule_update"] = strategy_rule_update
    _save_settlement_state(state)
    dashboard = build_dashboard_summary()
    return {
        "status": "ok",
        "settled_count": settled_count,
        "pending_count": pending_count,
        "skipped_count": skipped_count,
        "horizons": [f"T+{x}" for x in selected_horizons],
        "evaluation": state["evaluation"],
        "strategy_rule_update": strategy_rule_update,
        "dashboard_updated_at": dashboard.get("updated_at"),
    }
