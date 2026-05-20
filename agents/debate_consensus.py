"""双 AI 观点对立博弈 — 合作纳什共识求解（非零和）。"""

from __future__ import annotations

import re
from typing import Any


def _sentiment_side(case: dict) -> str:
    s = str(case.get("sentiment", "neutral") or "neutral").lower()
    if s in {"positive", "bull", "buy"}:
        return "pro"
    if s in {"negative", "bear", "sell"}:
        return "con"
    bull = float(case.get("bull_ratio", 0.5) or 0.5)
    bear = float(case.get("bear_ratio", 0.5) or 0.5)
    if bull > bear + 0.08:
        return "pro"
    if bear > bull + 0.08:
        return "con"
    return "neutral"


def split_pro_con_cases(case_a: dict, case_b: dict) -> tuple[dict, dict, dict, dict]:
    """按多空立场划分赞成/反对智能体，返回 pro_case, con_case, pro_label, con_label。"""
    side_a = _sentiment_side(case_a)
    side_b = _sentiment_side(case_b)
    if side_a == "pro" and side_b != "pro":
        return case_a, case_b, case_a, case_b
    if side_b == "pro" and side_a != "pro":
        return case_b, case_a, case_b, case_a
    if side_a == "con" and side_b != "con":
        return case_b, case_a, case_b, case_a
    if side_b == "con" and side_a != "con":
        return case_a, case_b, case_a, case_b
    # 中性对峙：A 偏技术资金视为短线落地，B 偏基本面新闻视为价值/风险
    bull_a = float(case_a.get("bull_ratio", 0.5) or 0.5)
    bull_b = float(case_b.get("bull_ratio", 0.5) or 0.5)
    if bull_a >= bull_b:
        return case_a, case_b, case_a, case_b
    return case_b, case_a, case_b, case_a


def build_agent_profile(case: dict, role: str = "pro") -> dict:
    """提取核心论据、收益点、风险点、落地条件。"""
    reasoning = str(case.get("reasoning", "") or "")
    thought = str(case.get("thought_process", "") or "")
    text = f"{reasoning} {thought}".strip()
    benefits: list[str] = []
    risks: list[str] = []
    conditions: list[str] = []
    for seg in re.split(r"[|；;\n]+", text):
        seg = seg.strip()
        if not seg:
            continue
        low = seg.lower()
        if any(k in seg for k in ("风险", "回撤", "利空", "压制", "规避", "谨慎", "破位", "流出", "高估")):
            risks.append(seg[:120])
        elif any(k in seg for k in ("若", "待", "需", "确认", "企稳", "观察", "条件", "催化")):
            conditions.append(seg[:120])
        elif any(k in seg for k in ("机会", "优势", "利好", "修复", "低估", "流入", "突破", "安全边际", "看多")):
            benefits.append(seg[:120])
        elif role == "pro":
            benefits.append(seg[:120])
        else:
            risks.append(seg[:120])

    if not benefits and role == "pro":
        benefits.append(reasoning[:120] or "赞成方未给出明确收益点")
    if not risks and role == "con":
        risks.append(reasoning[:120] or "反对方未给出明确风险点")
    if not conditions:
        conditions.append("需结合盘面与资金确认后再执行")

    return {
        "agent": case.get("agent", "未知分析师"),
        "stance": role,
        "sentiment": case.get("sentiment", "neutral"),
        "confidence": float(case.get("confidence", 0.5) or 0.5),
        "bull_ratio": float(case.get("bull_ratio", 0.5) or 0.5),
        "bear_ratio": float(case.get("bear_ratio", 0.5) or 0.5),
        "core_thesis": reasoning[:200] or thought[:200],
        "benefits": benefits[:4],
        "risks": risks[:4],
        "conditions": conditions[:3],
    }


def infer_scenario_weights(pro_profile: dict, con_profile: dict) -> tuple[float, float, str]:
    """场景权重：偏落地实操 / 偏风险管控 / 中性均等。"""
    pro_conf = float(pro_profile.get("confidence", 0.5) or 0.5)
    con_conf = float(con_profile.get("confidence", 0.5) or 0.5)
    con_risk_n = len(con_profile.get("risks", []) or [])
    pro_benefit_n = len(pro_profile.get("benefits", []) or [])

    if con_conf >= pro_conf + 0.12 or con_risk_n >= pro_benefit_n + 2:
        return 0.4, 0.6, "risk_control"
    if pro_conf >= con_conf + 0.12 or pro_benefit_n >= con_risk_n + 2:
        return 0.6, 0.4, "execution_focus"
    return 0.5, 0.5, "neutral"


def _strategy_balance_score(pro_util: float, con_util: float) -> float:
    """合作博弈效用：双方均部分成立时最高，单边极端最低。"""
    balance = 1.0 - abs(pro_util - con_util)
    floor_util = min(pro_util, con_util)
    return round(balance * 0.55 + floor_util * 0.45, 4)


def compute_payoff_matrix(
    pro_profile: dict,
    con_profile: dict,
    pro_weight: float,
    con_weight: float,
) -> dict[str, Any]:
    """策略：全盘赞成 / 全盘反对 / 融合共识。"""
    pro_strength = float(pro_profile.get("confidence", 0.5) or 0.5) * float(pro_profile.get("bull_ratio", 0.5) or 0.5)
    con_strength = float(con_profile.get("confidence", 0.5) or 0.5) * float(con_profile.get("bear_ratio", 0.5) or 0.5)
    pro_strength = max(0.05, min(1.0, pro_strength))
    con_strength = max(0.05, min(1.0, con_strength))

    full_pro = {"pro_util": round(pro_strength, 3), "con_util": round(1.0 - con_strength, 3)}
    full_pro["combined_utility"] = _strategy_balance_score(full_pro["pro_util"], full_pro["con_util"])
    full_pro["label"] = "全盘赞成"

    full_con = {"pro_util": round(1.0 - pro_strength, 3), "con_util": round(con_strength, 3)}
    full_con["combined_utility"] = _strategy_balance_score(full_con["pro_util"], full_con["con_util"])
    full_con["label"] = "全盘反对"

    fusion_pro = min(0.85, 0.45 + 0.35 * pro_weight + 0.15 * pro_strength)
    fusion_con = min(0.85, 0.45 + 0.35 * con_weight + 0.15 * con_strength)
    fusion = {"pro_util": round(fusion_pro, 3), "con_util": round(fusion_con, 3)}
    fusion["combined_utility"] = _strategy_balance_score(fusion["pro_util"], fusion["con_util"])
    fusion["label"] = "融合共识"

    scores = {
        "full_pro": full_pro["combined_utility"],
        "full_con": full_con["combined_utility"],
        "fusion": fusion["combined_utility"],
    }
    best_key = max(scores, key=scores.get)

    return {
        "full_pro": full_pro,
        "full_con": full_con,
        "fusion": fusion,
        "best_strategy": best_key,
        "scores": scores,
    }


def build_alignment_bundle(ticker: str, case_a: dict, case_b: dict) -> dict[str, Any]:
    pro_case, con_case, pro_raw, con_raw = split_pro_con_cases(case_a, case_b)
    pro_profile = build_agent_profile(pro_case, role="pro")
    con_profile = build_agent_profile(con_case, role="con")
    pro_weight, con_weight, scenario = infer_scenario_weights(pro_profile, con_profile)
    matrix = compute_payoff_matrix(pro_profile, con_profile, pro_weight, con_weight)
    return {
        "ticker": ticker,
        "pro_case": pro_case,
        "con_case": con_case,
        "pro_profile": pro_profile,
        "con_profile": con_profile,
        "pro_weight": pro_weight,
        "con_weight": con_weight,
        "scenario": scenario,
        "payoff_matrix": matrix,
        "decision_goal": "在统一风控约束下形成可执行的多空共识，避免单边极端",
        "criteria": "事实可验证论据保留；主观臆断与情绪化极端剔除；风险边界与价值路径双向绑定",
    }


def rule_based_consensus(alignment: dict) -> dict[str, Any]:
    """矩阵 + 权重规则兜底，输出合作博弈共识。"""
    pro_p = alignment["pro_profile"]
    con_p = alignment["con_profile"]
    matrix = alignment["payoff_matrix"]
    pro_w = alignment["pro_weight"]
    con_w = alignment["con_weight"]
    best = matrix["best_strategy"]

    pro_kept = "；".join((pro_p.get("benefits") or [])[:2]) or pro_p.get("core_thesis", "")
    con_kept = "；".join((con_p.get("risks") or [])[:2]) or con_p.get("core_thesis", "")
    conditions = "；".join((pro_p.get("conditions") or [])[:1] + (con_p.get("conditions") or [])[:1])

    if best == "fusion" or alignment["scenario"] == "neutral":
        mode = "cooperative_consensus"
        bull = float(pro_p.get("bull_ratio", 0.5) or 0.5) * pro_w
        bear = float(con_p.get("bear_ratio", 0.5) or 0.5) * con_w
        if bull > bear * 1.08:
            decision = "BUY"
        elif bear > bull * 1.08:
            decision = "SELL"
        else:
            decision = "HOLD"
        confidence = round(min(0.85, (float(pro_p.get("confidence", 0.5)) * pro_w + float(con_p.get("confidence", 0.5)) * con_w)), 3)
        reasoning = (
            f"合作博弈收敛为融合共识：保留赞成方可行优势（{pro_kept[:60]}），"
            f"纳入反对方风险边界（{con_kept[:60]}），剔除极端单边。"
        )
    elif best == "full_pro":
        mode = "expectation_gap" if float(pro_p.get("confidence", 0.5)) >= 0.65 else "cooperative_consensus"
        decision = "BUY"
        confidence = float(pro_p.get("confidence", 0.5) or 0.5)
        reasoning = f"博弈矩阵偏向全盘赞成，但纳入反方风险约束后轻仓执行：{con_kept[:50]}"
    else:
        mode = "cooperative_consensus"
        decision = "HOLD" if alignment["scenario"] == "risk_control" else "SELL"
        confidence = float(con_p.get("confidence", 0.5) or 0.5)
        reasoning = f"博弈矩阵偏风险管控，执行以防守为主：{con_kept[:60]}"

    return {
        "mode": mode,
        "decision": decision,
        "confidence": confidence,
        "fusion_strategy": matrix[best]["label"],
        "game_matrix": {
            "full_pro": matrix["full_pro"],
            "full_con": matrix["full_con"],
            "fusion": matrix["fusion"],
        },
        "consensus_layers": {
            "argument_clash": f"事实论据保留，冲突点取平衡；赞成权重={pro_w:.2f}，反对权重={con_w:.2f}",
            "constraint_binding": f"赞成方接纳风险边界：{con_kept[:80]}；反对方接纳价值路径：{pro_kept[:80]}",
            "weight_balance": f"场景={alignment['scenario']}，权重 pro={pro_w:.2f} / con={con_w:.2f}",
        },
        "pro_content_kept": pro_kept,
        "con_content_kept": con_kept,
        "extreme_removed": "剔除无证据单边满仓/空仓与情绪化臆断",
        "pro_weight": pro_w,
        "con_weight": con_w,
        "reasoning": reasoning,
        "next_action": conditions or "按融合共识进入风控仓位计算",
        "wrong_agent": "",
    }


def cooperative_judge_prompt(ticker: str, alignment: dict, round_idx: int = 0) -> str:
    pro_p = alignment["pro_profile"]
    con_p = alignment["con_profile"]
    matrix = alignment["payoff_matrix"]
    return (
        "你是【双AI观点对立博弈裁判】。博弈类型：合作博弈（非零和），目标是剔除极端立场，收敛为统一共识最优方案。\n"
        f"标的：{ticker} | 轮次：{round_idx}\n"
        f"统一目标：{alignment['decision_goal']}\n"
        f"评判标准：{alignment['criteria']}\n\n"
        "【信息对齐】\n"
        f"赞成策略Agent（{pro_p['agent']}）：方向={pro_p['sentiment']}，置信={pro_p['confidence']:.2f}，"
        f"核心论据={pro_p['core_thesis']}\n"
        f"  - 收益点：{' | '.join(pro_p.get('benefits', [])[:3]) or '无'}\n"
        f"  - 落地条件：{' | '.join(pro_p.get('conditions', [])[:2]) or '无'}\n"
        f"反对策略Agent（{con_p['agent']}）：方向={con_p['sentiment']}，置信={con_p['confidence']:.2f}，"
        f"核心论据={con_p['core_thesis']}\n"
        f"  - 风险点：{' | '.join(con_p.get('risks', [])[:3]) or '无'}\n"
        f"  - 约束条件：{' | '.join(con_p.get('conditions', [])[:2]) or '无'}\n\n"
        "【收益博弈矩阵（已预计算参考）】\n"
        f"- 全盘赞成：赞成效用={matrix['full_pro']['pro_util']}，反对效用={matrix['full_pro']['con_util']}，"
        f"综合={matrix['full_pro']['combined_utility']}（通常失衡）\n"
        f"- 全盘反对：赞成效用={matrix['full_con']['pro_util']}，反对效用={matrix['full_con']['con_util']}，"
        f"综合={matrix['full_con']['combined_utility']}（通常失衡）\n"
        f"- 融合共识：赞成效用={matrix['fusion']['pro_util']}，反对效用={matrix['fusion']['con_util']}，"
        f"综合={matrix['fusion']['combined_utility']}（合作纳什最优）\n"
        f"场景权重：{alignment['scenario']}，赞成权重={alignment['pro_weight']:.2f}，反对权重={alignment['con_weight']:.2f}\n\n"
        "【三层博弈收敛规则】\n"
        "1. 论据对抗：事实成立保留，主观臆断剔除；\n"
        "2. 约束绑定：赞成方必须接纳反对方风险防控边界，反对方必须接纳赞成方价值落地路径；\n"
        "3. 权重均衡：按场景分配双方权重。\n\n"
        "【共识公式】统一建议 = 赞成方合理可行内容 + 反对方风险规避内容 - 双方极端偏执内容\n\n"
        "只输出合法 JSON："
        '{"mode":"cooperative_consensus/expectation_gap/opponent_wrong/unclear",'
        '"decision":"BUY/SELL/HOLD","confidence":0.0到1.0,'
        '"fusion_strategy":"全盘赞成/全盘反对/融合共识",'
        '"game_matrix":{"full_pro":{},"full_con":{},"fusion":{}},'
        '"consensus_layers":{"argument_clash":"论据对抗结论",'
        '"constraint_binding":"约束绑定结论","weight_balance":"权重均衡说明"},'
        '"pro_content_kept":"保留的赞成内容","con_content_kept":"保留的反对/风险内容",'
        '"extreme_removed":"剔除的极端内容","pro_weight":0.0到1.0,"con_weight":0.0到1.0,'
        '"wrong_agent":"若一方明显错误填agent名否则空","reasoning":"120字内统一共识",'
        '"next_action":"下一步执行建议"}'
    )
