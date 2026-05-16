from .base import BaseAgent
import numpy as np
import time
from dataflows.providers.akshare_provider import AkShareProvider
import json
import os
from rl.reward import compute_trade_reward

provider = AkShareProvider()

# 加载结构化角色定义
ROLE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "json", "roles.json")
try:
    with open(ROLE_CONFIG_PATH, "r", encoding="utf-8") as f:
        ROLES_CONFIG = json.load(f)
except FileNotFoundError:
    ROLES_CONFIG = {}

def parse_llm_json(llm_result: str):
    """尝试将大模型返回解析为结构化的JSON字典数据"""
    raw = str(llm_result or "")
    try:
        # 清理可能包含的markdown json代码块标记
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        if not cleaned.startswith("{"):
            import re
            m = re.search(r"\{[\s\S]*\}", cleaned)
            if m:
                cleaned = m.group(0).strip()
        parsed = json.loads(cleaned.strip())
        if isinstance(parsed, dict):
            parsed["_parse_ok"] = True
            return parsed
        raise ValueError("parsed json is not object")
    except Exception as e:
        # Fallback 策略
        sentiment = "neutral"
        if "positive" in raw.lower(): sentiment = "positive"
        elif "negative" in raw.lower(): sentiment = "negative"
        return {
            "sentiment": sentiment,
            "reasoning": raw,
            "thought_process": "解析失败，无法获取思维链",
            "confidence": 0.5,
            "_parse_ok": False,
            "_parse_error": str(e),
        }


def normalize_decision(value: str) -> str:
    if value is None:
        return ""

    token = str(value).upper().strip()
    if "BUY" in token or "做多" in token:
        return "BUY"
    if "SELL" in token or "做空" in token:
        return "SELL"
    if "HOLD" in token or "观望" in token or "中性" in token:
        return "HOLD"
    return ""


def build_report_digest(reports: list) -> tuple[str, float, float]:
    role_weights = {
        "技术面分析师": 1.0,
        "舆情分析师": 0.9,
        "基本面分析师": 1.25,
        "宏观分析师": 1.15,
        "主力资金分析师": 1.05,
        "新闻研报专家": 1.2,
        "深度学习量化专家": 1.25,
    }

    lines = []
    bull_score = 0.0
    bear_score = 0.0

    for report in reports:
        sentiment = str(report.get("sentiment", "neutral")).lower()
        confidence = float(report.get("confidence", 0.5) or 0.5)
        agent_name = report.get("agent", "未知分析师")
        weight = role_weights.get(agent_name, 1.0)
        weighted_score = round(confidence * weight, 3)
        reasoning = report.get("reasoning", "")

        if sentiment == "positive":
            bull_score += weighted_score
        elif sentiment == "negative":
            bear_score += weighted_score

        lines.append(
            f"- {agent_name} | sentiment={sentiment} | confidence={confidence:.2f} | role_weight={weight:.2f} | weighted={weighted_score:.2f} | reason={reasoning}"
        )

    digest = "\n".join(lines) if lines else "暂无可用分析报告。"
    return digest, round(bull_score, 2), round(bear_score, 2)


def calculate_trend_strength(bull_score: float, bear_score: float) -> float:
    """用多空分差占总分比例衡量趋势强度，0 表示震荡，1 表示单边。"""
    total_score = max(float(bull_score or 0.0) + float(bear_score or 0.0), 0.0)
    if total_score <= 0:
        return 0.0
    return abs(float(bull_score or 0.0) - float(bear_score or 0.0)) / total_score


def has_score_advantage(leader_score: float, laggard_score: float, threshold: float = 0.3) -> bool:
    """判断领先方是否比落后方至少高出指定比例。"""
    leader = float(leader_score or 0.0)
    laggard = float(laggard_score or 0.0)
    if leader <= 0:
        return False
    if laggard <= 0:
        return True
    return leader >= laggard * (1.0 + threshold)


def sentiment_to_decision(sentiment: str) -> str:
    token = str(sentiment or "neutral").lower().strip()
    if token == "positive":
        return "BUY"
    if token == "negative":
        return "SELL"
    return "HOLD"


def normalize_ratio_case(case: dict) -> dict:
    bull_ratio = float(case.get("bull_ratio", 0.5) or 0.5)
    bear_ratio = float(case.get("bear_ratio", 0.5) or 0.5)
    total = bull_ratio + bear_ratio
    if total <= 0:
        bull_ratio, bear_ratio = 0.5, 0.5
    elif total > 1.01 or total < 0.99:
        bull_ratio, bear_ratio = bull_ratio / total, bear_ratio / total

    sentiment = case.get("sentiment")
    if not sentiment:
        if bull_ratio > bear_ratio:
            sentiment = "positive"
        elif bear_ratio > bull_ratio:
            sentiment = "negative"
        else:
            sentiment = "neutral"

    confidence = case.get("confidence")
    if confidence is None:
        confidence = abs(bull_ratio - bear_ratio)

    case["bull_ratio"] = round(max(0.0, min(1.0, bull_ratio)), 3)
    case["bear_ratio"] = round(max(0.0, min(1.0, bear_ratio)), 3)
    case["sentiment"] = str(sentiment).lower()
    case["confidence"] = float(confidence or 0.5)
    return case

# ==========================================
# 1. 数据与基础分析师团队 (Analysts)
# ==========================================

class TechnicalAnalyst(BaseAgent):
    def __init__(self, name: str):
        role_name = ROLES_CONFIG.get("TechnicalAnalyst", {}).get("role", "技术面分析师")
        super().__init__(name, role_name)
        self.config = ROLES_CONFIG.get("TechnicalAnalyst", {})

    def step(self, ticker: str, features: np.ndarray, target_date: str = None):
        self.log(f"分析 [{ticker}] 的K线形态、均线与动量指标...")
        data_str = "暂无足够技术面数据"
        if features is not None and len(features) > 0:
            last_few_days = features[-5:] if len(features) > 5 else features
            data_str = f"过去几天的数据特征张量 (例如标准化后的开盘、收盘等): {np.round(last_few_days, 4).tolist()}"

        self.log(f"✅ 获取到技术面输入数据: {data_str[:150]}...")
        prompt_template = self.config.get(
            "prompt_template", 
            "你是一个专业的技术面量化分析师。针对股票 {ticker}，以下是它最新截面的量价技术面数据：\n{data}\n请判断技术面当前呈现出的涨跌倾向。请在回答最后明确包含 'positive', 'negative'，或 'neutral' 中的一个英文单词代表情绪。你的理由尽量简短(50字以内)。"
        )
        prompt = prompt_template.format(ticker=ticker, data=data_str)
        llm_result = self.query_llm(prompt)
        self.log(f"LLM根据技术面数据推理得出: {llm_result}")
        
        parsed = parse_llm_json(llm_result)
        sentiment = parsed.get("sentiment", "neutral")
        reasoning = parsed.get("reasoning", llm_result)
        thought_process = parsed.get("thought_process", "无")
        confidence = parsed.get("confidence", 0.5)
        
        return {"agent": self.name, "sentiment": sentiment, "confidence": confidence, "reasoning": reasoning, "thought_process": thought_process}

class SentimentAnalyst(BaseAgent):
    def __init__(self, name: str):
        role_name = ROLES_CONFIG.get("SentimentAnalyst", {}).get("role", "舆情分析师")
        super().__init__(name, role_name)
        self.config = ROLES_CONFIG.get("SentimentAnalyst", {})

    def step(self, ticker: str, target_date: str = None):
        self.log(f"挖掘 [{ticker}] 社交媒体(股吧、雪球等)新闻散户情绪...")
        # 避免未来函数：传入 target_date 作为截止日期，provider 会过滤掉晚于 target_date 的新闻
        news_str = provider.fetch_sentiment_data(ticker, cutoff_date=target_date)
        self.log(f"✅ 成功抓取舆情: {news_str}")

        prompt_template = self.config.get(
            "prompt_template",
            "你是一名专门对接对冲基金的散户舆情与新闻情感分析师。对于股票代码 {ticker}，相关市场舆情如下：\n{data}\n请你研判整体散户与新闻面传递出的情绪是多头还是空头。请在回答段落最后明确输出 'positive', 'negative'，或 'neutral'。附带简短推理逻辑(50字以内)。"
        )
        prompt = prompt_template.format(ticker=ticker, data=news_str)
        llm_result = self.query_llm(prompt)
        self.log(f"LLM根据舆情数据推理得出: {llm_result}")
        
        parsed = parse_llm_json(llm_result)
        sentiment = parsed.get("sentiment", "neutral")
        reasoning = parsed.get("reasoning", llm_result)
        thought_process = parsed.get("thought_process", "无")
        confidence = parsed.get("confidence", 0.5)
        
        return {"agent": self.name, "sentiment": sentiment, "confidence": confidence, "reasoning": reasoning, "thought_process": thought_process}

class FundamentalAnalyst(BaseAgent):
    def __init__(self, name: str):
        role_name = ROLES_CONFIG.get("FundamentalAnalyst", {}).get("role", "基本面分析师")
        super().__init__(name, role_name)
        self.config = ROLES_CONFIG.get("FundamentalAnalyst", {})

    def step(self, ticker: str, target_date: str = None):
        self.log(f"分析 [{ticker}] 财报数据(PE, PB)及行业研报估值...")
        data_str = provider.fetch_fundamental_data(ticker, cutoff_date=target_date)
        self.log(f"✅ 成功抓取基本面核心估值数据: {data_str}")

        prompt_template = self.config.get(
            "prompt_template",
            "你是一名资深的价值投资基本面分析师。针对股票 {ticker}，以下是最新的真实基本面估值数据：\n{data}\n结合该行业的普遍情况与估值分布（如PE/PB是否具备安全边际），研判当前基本面健康度。请在结尾明确包含 'positive', 'negative' 或 'neutral'。 给出极简分析逻辑(50字内)。"
        )
        prompt = prompt_template.format(ticker=ticker, data=data_str)
        llm_result = self.query_llm(prompt)
        self.log(f"LLM根据基本面数据推理得出: {llm_result}")
        
        parsed = parse_llm_json(llm_result)
        sentiment = parsed.get("sentiment", "neutral")
        reasoning = parsed.get("reasoning", llm_result)
        thought_process = parsed.get("thought_process", "无")
        confidence = parsed.get("confidence", 0.5)
            
        return {"agent": self.name, "sentiment": sentiment, "confidence": confidence, "reasoning": reasoning, "thought_process": thought_process}

class MacroAnalyst(BaseAgent):
    def __init__(self, name: str):
        role_name = ROLES_CONFIG.get("MacroAnalyst", {}).get("role", "宏观经济分析师")
        super().__init__(name, role_name)
        self.config = ROLES_CONFIG.get("MacroAnalyst", {})

    def step(self, ticker: str, target_date: str = None):
        self.log(f"评估宏观周期、利率环境及大盘(上证指数)系统性风险...")
        macro_str = provider.fetch_macro_data(cutoff_date=target_date)
        self.log(f"✅ 成功抓取近期上证大盘走势: {macro_str[:50]}...")

        prompt_template = self.config.get(
            "prompt_template",
            "你是一名宏观经济及大盘系统性风险分析师。结合以下A股上证大盘近期的真实指标：\n{data}\n请判断当前市场整体系统性环境、流动性情绪对做多个股是否具备支撑。回答结尾必须明确输出 'positive', 'negative' 或 'neutral'，理由需非常精简(不超过50字) 。"
        )
        prompt = prompt_template.format(ticker=ticker, data=macro_str)
        llm_result = self.query_llm(prompt)
        self.log(f"LLM根据宏观大盘数据推理得出: {llm_result}")
        
        parsed = parse_llm_json(llm_result)
        sentiment = parsed.get("sentiment", "neutral")
        reasoning = parsed.get("reasoning", llm_result)
        thought_process = parsed.get("thought_process", "无")
        confidence = parsed.get("confidence", 0.5)
            
        return {"agent": self.name, "sentiment": sentiment, "confidence": confidence, "reasoning": reasoning, "thought_process": thought_process}

class SmartMoneyAnalyst(BaseAgent):
    def __init__(self, name: str):
        role_name = ROLES_CONFIG.get("SmartMoneyAnalyst", {}).get("role", "主力资金分析师")
        super().__init__(name, role_name)
        self.config = ROLES_CONFIG.get("SmartMoneyAnalyst", {})

    def step(self, ticker: str, target_date: str = None):
        self.log(f"监控 [{ticker}] 北向资金、机构龙虎榜与大单净流入...")
        flow_str = provider.fetch_smart_money_data(ticker, cutoff_date=target_date)
        self.log(f"✅ 成功提取主力大单资金净流入: {flow_str[:50]}...")

        prompt_template = self.config.get(
            "prompt_template",
            "你是量化团队中的主力游资（Smart Money）追踪监测分析师。对于股票 {ticker}，以下是你捕获到的最新主力净流入异动数据：\n{data}\n请判断游资或机构主力目前是在洗盘吸筹、出货派发还是观望。最后一行必须包含 'positive', 'negative' 或 'neutral'。 理由请控制在50字左右。"
        )
        prompt = prompt_template.format(ticker=ticker, data=flow_str)
        llm_result = self.query_llm(prompt)
        self.log(f"LLM根据主力资金数据推理得出: {llm_result}")
        
        parsed = parse_llm_json(llm_result)
        sentiment = parsed.get("sentiment", "neutral")
        reasoning = parsed.get("reasoning", llm_result)
        thought_process = parsed.get("thought_process", "无")
        confidence = parsed.get("confidence", 0.5)
            
        return {"agent": self.name, "sentiment": sentiment, "confidence": confidence, "reasoning": reasoning, "thought_process": thought_process}

class NewsAnalystAgent(BaseAgent):
    def __init__(self, name: str, rag_engine):
        super().__init__(name, "新闻研报特工 (Agentic RAG)")
        self.rag = rag_engine
        self.config = ROLES_CONFIG.get("NewsAnalystAgent", {})

    def step(self, ticker: str, target_date: str = None):
        date_str = f"({target_date})" if target_date else ""
        self.log(f"🔎 启动 Agentic RAG 检索迭代特工，目标标的: [{ticker}] {date_str}...")
        
        max_iters = 3
        # 简化检索词，避免太长导致匹配不到
        base_date = str(target_date)[:10] if target_date else ""
        if base_date:
            try:
                import datetime
                dt = datetime.datetime.strptime(base_date, "%Y-%m-%d")
                quarter = (dt.month - 1) // 3 + 1
                q_str = "一季度" if quarter == 1 else f"{quarter}季度"
                query = f"{ticker} {dt.year}{q_str}"
            except Exception:
                query = f"{ticker} {base_date}"
        else:
            query = ticker

        all_docs = []
        information_sufficient = False
        
        # Agentic RAG 反思检索循环
        for i in range(max_iters):
            self.log(f"  -> 第 {i+1} 轮检索, 搜索词: '{query}'")
            docs = self.rag.retrieve(query, target_date=target_date, ticker=ticker, top_k=2)
            if docs:
                all_docs.extend(docs)
                
            # 去重
            all_docs = list(set(all_docs))
            if not all_docs:
                self.log("  -> 未检索到任何有效外部文本，直接退出。")
                break
                
            summary = "\n".join(all_docs)
            eval_prompt = self.config.get(
                "eval_prompt_template",
                "你是一个智能搜索研报与新闻的特工。我们正在分析股票 【{ticker}】。目前我们收集到的情报如下：\n{summary}\n请判断这些信息是否足够支撑趋势预判？返回合法 JSON：{{\"enough\": true或false, \"next_query\": \"如果不够请给更精简干练的词(例如 '{ticker} 研报')\", \"reason\": \"50字内理由\"}}"
            ).format(ticker=ticker, summary=summary)
            
            try:
                eval_result = self.query_llm(eval_prompt)
                parsed_eval = parse_llm_json(eval_result)
                is_enough = parsed_eval.get("enough", True)
                reason = parsed_eval.get("reason", "无法解析评估理由。")
                
                self.log(f"  -> RAG 评估结果: 足够={is_enough}, 理由: {reason}")
                
                if is_enough:
                    information_sufficient = True
                    break
                else:
                    query = parsed_eval.get("next_query", f"{ticker} 研报")
                    if not query or query == ticker:
                        break
            except Exception as e:
                self.log(f"  -> Agentic 评估环节出错，中断检索循环: {e}")
                break

        if not all_docs:
             self.log("  -> RAG 检索全空，拒绝瞎猜，返回中立判定。")
             return {"agent": self.name, "sentiment": "neutral", "confidence": 0.1, "reasoning": "未检索到任何近期有效新闻研报，信息不足。", "thought_process": "信息不足，保持中立以免产生幻觉。"}

        low_info_mode = not information_sufficient
        if low_info_mode:
            self.log("  -> 多轮RAG评估均为信息不足，进入低置信度偏好模式。")
            
        final_summary = "总体文献摘要：\n" + "\n".join(all_docs)
        self.scratchpad.append(f"Agentic RAG 工作流结束。收集到文献数为: {len(all_docs)}")
        
        # 最终汇总判定
        if low_info_mode:
            final_prompt = self.config.get(
                "final_prompt_template_low_info",
                "你是一名资深量化系统的【新闻研报分析师】。针对标的【{ticker}】，当前证据完整性不足，但仍需要输出一个轻微方向偏好。\n已检索到的文献切片如下：\n{final_summary}\n要求：1) sentiment 必须为 positive 或 negative（二选一，不要 neutral）；2) confidence 必须在 0.20~0.35；3) reasoning 需明确提示“证据不足，仅作轻微倾向”。只输出合法 JSON：{{\"thought_process\": \"简要推理\", \"sentiment\": \"positive或negative\", \"confidence\": 0.2到0.35, \"reasoning\": \"一句话结论\"}}"
            ).format(ticker=ticker, final_summary=final_summary)
        else:
            final_prompt = self.config.get(
                "final_prompt_template",
                "你是一名资深量化系统的【新闻研报分析师】。针对标的 【{ticker}】，经过多轮 Agentic 检索获取到的全部相关深度切片如下：\n{final_summary}\n请你判断新闻与基本面传递出的综合情绪，并给出一个明确的看多、看空、或者中性评级。务必只输出合法的纯 JSON 格式：{{\"thought_process\": \"基于这些深度文献的归纳梳理和逻辑判断链条\", \"sentiment\": \"必须是 positive, negative 或 neutral 之一\", \"confidence\": 0.0到1.0的浮点数, \"reasoning\": \"一句话精炼你的结论(50字内)\"}}"
            ).format(ticker=ticker, final_summary=final_summary)
        
        llm_conclusion = self.query_llm(final_prompt)
        parsed_final = parse_llm_json(llm_conclusion)
        
        sentiment = parsed_final.get("sentiment", "neutral")
        reasoning = parsed_final.get("reasoning", llm_conclusion)
        thought_process = parsed_final.get("thought_process", "提取失败。")
        confidence = parsed_final.get("confidence", 0.5)

        if low_info_mode:
            sentiment_norm = str(sentiment).lower().strip()
            if sentiment_norm not in {"positive", "negative"}:
                # 证据不足场景下采用保守风险偏好：无法解析时默认 negative（轻微偏空）
                sentiment_norm = "negative"
            sentiment = sentiment_norm
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.25
            confidence = max(0.2, min(0.35, confidence))
            reasoning = f"{reasoning}（证据不足，仅作轻微倾向）"
        
        self.log(f"✅ RAG 特工最终裁决: {sentiment} (置信度:{confidence})")
        return {"agent": self.name, "sentiment": sentiment, "confidence": float(confidence), "reasoning": reasoning, "thought_process": thought_process}

class QuantResearcherAgent(BaseAgent):
    def __init__(self, name: str, dl_engine):
        super().__init__(name, "深度学习量化研究员(Hybrid AI)")
        self.dl = dl_engine
        self.config = ROLES_CONFIG.get("QuantResearcherAgent", {})

    def step(self, ticker: str, features_override: np.ndarray = None, target_date: str = None):
        self.log(f"输入张量特征执行 LSTM 深度模型推演...")
        if features_override is None:
            features = np.random.rand(10, 10)
        else:
            features = features_override

        # 1. 深度学习得出纯数值
        pred = self.dl.predict(ticker, features)
        
        dl_score = pred["score"]
        dl_confidence_str = pred.get("confidence", "50.0%").replace("%", "")

        try:
            dl_confidence = min(float(dl_confidence_str) / 100.0, 1.0)
        except:
            dl_confidence = 0.5

        trend = "看多" if dl_score > 0 else "看空"

        self.log(f"LSTM 预测 (无情绪纯数学打分): {pred.get('trend','')} 原始分[{dl_score:.4f}] 预测置信度[{dl_confidence:.2f}]")

        # 置信度过滤：若置信度过低则直接弃用该预测，避免无效模型输出影响决策
        CONF_THRESHOLD = float(self.config.get("confidence_threshold", 0.3))
        if dl_confidence < CONF_THRESHOLD:
            self.log(f"LSTM预测置信度不足 ({dl_confidence:.2f} < {CONF_THRESHOLD}), 已过滤，不参与多空评分。")
            return {"agent": self.name, "sentiment": "neutral", "confidence": 0.0, "reasoning": "LSTM置信度不足，已过滤", "thought_process": "置信度过滤"}
        
        # 2. 让 LLM 解释深度学习模型输出 (多模态 / Hybrid AI)
        prompt = self.config.get(
            "prompt_template",
            "你是一名跨模态量化分析师，专门负责将深度学习 (DL) 模型的冰冷回测数值转化为可读的交易逻辑。现在是针对股票 {ticker} 的深度学习分析：\n- 底层模型: LSTM 时间序列神经网络\n- 特征工程: 过去 10 个时间步的量价时序张量特征\n- 深度学习算力直接给出的预测值: {dl_score:.4f} (正数倾向于涨，负数倾向于跌)\n- 数学层面的置信度评估: {dl_confidence:.2%}\n请你结合算法化决策、能力圈和安全边际思维，判断这个预测值是否足够稳健。如果数学模型确定性不足，请明确指出局限性并倾向 HOLD。最终仅输出 JSON：{{\"thought_process\": \"你的跨模态融合解读思路\", \"sentiment\": \"必须是 positive, negative 或 neutral\", \"confidence\": 0.0到1.0的浮点数, \"reasoning\": \"简练的一句话量化预测总结(50字内)\"}}"
        ).format(ticker=ticker, dl_score=dl_score, dl_confidence=dl_confidence)
        llm_result = self.query_llm(prompt)
        parsed = parse_llm_json(llm_result)
        
        sentiment = parsed.get("sentiment", "positive" if dl_score > 0 else "negative")
        reasoning = parsed.get("reasoning", f"DL逻辑倾向 {trend} (打分:{dl_score:.2f})")
        confidence = parsed.get("confidence", dl_confidence)
        thought_process = parsed.get("thought_process", f"深度学习引擎计算的出趋势预测为{trend}。")
        
        return {"agent": self.name, "sentiment": sentiment, "confidence": float(confidence), "reasoning": reasoning, "thought_process": thought_process}


class CombinedAnalystAgent(BaseAgent):
    """把多个底层分析师合成为一个直接输出多空比率的研究员。"""

    def _build_ratio_case(self, ticker: str, reports: list, prompt_template: str) -> dict:
        evidence = "\n".join(
            [
                f"- {r.get('agent', '未知分析师')} | sentiment={r.get('sentiment')} | "
                f"confidence={float(r.get('confidence', 0.5) or 0.5):.2f} | {r.get('reasoning', '')}"
                for r in reports
            ]
        )
        prompt = prompt_template.format(ticker=ticker, evidence=evidence)
        llm_result = self.query_llm(prompt)
        parsed = parse_llm_json(llm_result)

        sentiment = parsed.get("sentiment")
        if not sentiment:
            sentiment = "positive" if "positive" in llm_result.lower() else ("negative" if "negative" in llm_result.lower() else "neutral")
        confidence = float(parsed.get("confidence", 0.5) or 0.5)
        bull_ratio = parsed.get("bull_ratio", parsed.get("bullish_ratio"))
        bear_ratio = parsed.get("bear_ratio", parsed.get("bearish_ratio"))
        if bull_ratio is None or bear_ratio is None:
            spread = max(0.1, min(0.45, abs(confidence - 0.5)))
            if str(sentiment).lower() == "positive":
                bull_ratio, bear_ratio = 0.5 + spread, 0.5 - spread
            elif str(sentiment).lower() == "negative":
                bull_ratio, bear_ratio = 0.5 - spread, 0.5 + spread
            else:
                bull_ratio, bear_ratio = 0.5, 0.5

        case = normalize_ratio_case({
            "agent": self.name,
            "sentiment": sentiment,
            "confidence": confidence,
            "bull_ratio": bull_ratio,
            "bear_ratio": bear_ratio,
            "reasoning": parsed.get("reasoning", llm_result),
            "thought_process": parsed.get("thought_process", parsed.get("analysis", llm_result)),
            "source_reports": reports,
        })
        self.log(
            f"综合输出: 多={case['bull_ratio']:.2f}, 空={case['bear_ratio']:.2f}, "
            f"方向={case['sentiment']}, 置信度={case['confidence']:.2f}"
        )
        return case

    def debate_round(self, ticker: str, own_case: dict, opponent_case: dict, referee_view: dict, round_idx: int) -> dict:
        prompt = (
            "你是 {agent}，正在与另一位分析师和裁判官进行第 {round_idx} 轮单独博弈。\n"
            "你的当前观点：多方比率={own_bull:.2f}, 空方比率={own_bear:.2f}, "
            "方向={own_sentiment}, 理由={own_reason}\n"
            "对方观点：多方比率={opp_bull:.2f}, 空方比率={opp_bear:.2f}, "
            "方向={opp_sentiment}, 理由={opp_reason}\n"
            "裁判临时判断：{referee_view}\n"
            "请判断你是否被说服，或者是否能更清晰说明对方错误。只输出合法 JSON："
            "{{\"sentiment\":\"positive/negative/neutral\", \"bull_ratio\":0.0到1.0, "
            "\"bear_ratio\":0.0到1.0, \"confidence\":0.0到1.0, "
            "\"conceded\":true或false, \"reasoning\":\"80字内更新后的观点\"}}"
        ).format(
            ticker=ticker,
            agent=self.name,
            round_idx=round_idx,
            own_bull=own_case.get("bull_ratio", 0.5),
            own_bear=own_case.get("bear_ratio", 0.5),
            own_sentiment=own_case.get("sentiment", "neutral"),
            own_reason=own_case.get("reasoning", ""),
            opp_bull=opponent_case.get("bull_ratio", 0.5),
            opp_bear=opponent_case.get("bear_ratio", 0.5),
            opp_sentiment=opponent_case.get("sentiment", "neutral"),
            opp_reason=opponent_case.get("reasoning", ""),
            referee_view=json.dumps(referee_view, ensure_ascii=False),
        )
        llm_result = self.query_llm(prompt)
        parsed = parse_llm_json(llm_result)
        updated = own_case.copy()
        updated.update({
            "sentiment": parsed.get("sentiment", own_case.get("sentiment", "neutral")),
            "confidence": parsed.get("confidence", own_case.get("confidence", 0.5)),
            "bull_ratio": parsed.get("bull_ratio", own_case.get("bull_ratio", 0.5)),
            "bear_ratio": parsed.get("bear_ratio", own_case.get("bear_ratio", 0.5)),
            "reasoning": parsed.get("reasoning", own_case.get("reasoning", "")),
            "conceded": bool(parsed.get("conceded", False)),
        })
        updated = normalize_ratio_case(updated)
        self.log(f"第 {round_idx} 轮更新观点: {updated['sentiment']} | {updated['reasoning']}")
        return updated


class TechnicalFlowAnalyst(CombinedAnalystAgent):
    def __init__(self, name: str):
        role_name = ROLES_CONFIG.get("TechnicalFlowAnalyst", {}).get("role", "技术资金综合分析师")
        super().__init__(name, role_name)
        self.technical = TechnicalAnalyst(name="技术面子模块")
        self.smart_money = SmartMoneyAnalyst(name="主力资金子模块")
        self.config = ROLES_CONFIG.get("TechnicalFlowAnalyst", {})

    def step(self, ticker: str, features: np.ndarray, target_date: str = None):
        tech_report = self.technical.step(ticker, features=features, target_date=target_date)
        flow_report = self.smart_money.step(ticker, target_date=target_date)
        prompt_template = self.config.get(
            "prompt_template",
            "你是【技术资金综合分析师】，负责把技术面趋势和主力资金行为合成一个交易方向。\n"
            "标的：{ticker}\n证据：\n{evidence}\n"
            "请输出你自己的多空比率判断，而不是简单复述子模块。只输出合法 JSON："
            "{{\"sentiment\":\"positive/negative/neutral\", \"bull_ratio\":0.0到1.0, "
            "\"bear_ratio\":0.0到1.0, \"confidence\":0.0到1.0, "
            "\"thought_process\":\"简短推理\", \"reasoning\":\"50字内结论\"}}"
        )
        return self._build_ratio_case(ticker, [tech_report, flow_report], prompt_template)


class FundamentalNewsAnalyst(CombinedAnalystAgent):
    def __init__(self, name: str, rag_engine):
        role_name = ROLES_CONFIG.get("FundamentalNewsAnalyst", {}).get("role", "基本面新闻综合分析师")
        super().__init__(name, role_name)
        self.fundamental = FundamentalAnalyst(name="基本面子模块")
        self.news = NewsAnalystAgent(name="新闻研报子模块", rag_engine=rag_engine)
        self.config = ROLES_CONFIG.get("FundamentalNewsAnalyst", {})

    def step(self, ticker: str, target_date: str = None):
        fund_report = self.fundamental.step(ticker, target_date=target_date)
        news_report = self.news.step(ticker, target_date=target_date)
        prompt_template = self.config.get(
            "prompt_template",
            "你是【基本面新闻综合分析师】，负责把估值质量、财报线索、新闻研报和预期变化合成一个交易方向。\n"
            "标的：{ticker}\n证据：\n{evidence}\n"
            "请输出你自己的多空比率判断，并特别说明是否存在预期差。只输出合法 JSON："
            "{{\"sentiment\":\"positive/negative/neutral\", \"bull_ratio\":0.0到1.0, "
            "\"bear_ratio\":0.0到1.0, \"confidence\":0.0到1.0, "
            "\"thought_process\":\"简短推理\", \"reasoning\":\"50字内结论\"}}"
        )
        return self._build_ratio_case(ticker, [fund_report, news_report], prompt_template)


# ==========================================
# 2. 辩论与博弈层 (Debate & Game Theory)
# ==========================================

class BullResearcher(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "看多逻辑辩手")
        self.config = ROLES_CONFIG.get("BullResearcher", {})

    def step(self, reports: list):
        bull_points = [r for r in reports if r['sentiment'] == 'positive']
        strength = sum(r.get('confidence', 1.0) for r in bull_points)
        self.log(f"提炼看多阵营子弹: 共收集 {len(bull_points)} 条看多逻辑，总置信度得分: {strength:.2f}。")
        
        if not bull_points:
            return {"agent": self.name, "strength": 0.0, "thesis": "暂无有效的看多逻辑支撑。"}
            
        info_str = "\n".join([f"- {r['agent']} (置信度:{r.get('confidence', 1.0)}): {r['reasoning']}" for r in bull_points])
        prompt = self.config.get(
            "prompt_template",
            "你目前代表【多方阵营】。这里是各分析师给出的看多信号：\n{info_str}\n请将它们凝聚成一篇精炼有力、具有说服力的多头辩论陈词(字数在100字以内)。请务必结合发出信号分析师的置信度进行表达（高置信度应更强硬）。"
        ).format(info_str=info_str)
        thesis = self.query_llm(prompt)
        self.log(f"看多辩词: {thesis}")
        return {"agent": self.name, "strength": round(strength, 2), "thesis": thesis}

    def cross_examine(self, my_case: dict, opponent_case: dict) -> dict:
        self.log(f"⚔️ 收到空方阵营的炮火，准备发起反击...")
        if opponent_case["strength"] == 0.0:
            return my_case # 空方无话可说，无需反驳
            
        prompt = self.config.get(
            "cross_exam_template",
            "你代表【多方阵营】参与量化多空辩论。你的初始阵地与逻辑是：\n{my_case}\n现在，【空方阵营】对市场提出了以下看空逻辑和威胁警告：\n{opponent_case}\n作为看多辩手，请你对空方的核心威胁进行一次犀利、客观的反驳(Cross-Examination)。如果空方理由确实致命，也要保留谨慎。请将【你的原逻辑】与【对空方的反驳】融合为一篇全新的、更加无懈可击的多头最终陈词 (字数严格要求在150字以内)。"
        ).format(my_case=my_case['thesis'], opponent_case=opponent_case['thesis'])
        new_thesis = self.query_llm(prompt)
        self.log(f"看多反击陈词: {new_thesis[:50]}...")
        return {"agent": self.name, "strength": my_case["strength"], "thesis": new_thesis}

class BearResearcher(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "看空逻辑辩手")
        self.config = ROLES_CONFIG.get("BearResearcher", {})

    def step(self, reports: list):
        bear_points = [r for r in reports if r['sentiment'] == 'negative']
        strength = sum(r.get('confidence', 1.0) for r in bear_points)
        self.log(f"提炼看空阵营子弹: 共收集 {len(bear_points)} 条看空逻辑，总置信度得分: {strength:.2f}。")
        
        if not bear_points:
            return {"agent": self.name, "strength": 0.0, "thesis": "暂无有效的看空逻辑支撑。"}
            
        info_str = "\n".join([f"- {r['agent']} (置信度:{r.get('confidence', 1.0)}): {r['reasoning']}" for r in bear_points])
        prompt = self.config.get(
            "prompt_template",
            "你目前代表【空方阵营】。这里是各分析师给出的看空或利空风险信号：\n{info_str}\n请将它们凝聚成一篇精炼有力、具有警示性的空头辩论陈词(字数在100字以内)。请务必结合发出信号分析师的置信度进行表达（高置信度应更强硬）。"
        ).format(info_str=info_str)
        thesis = self.query_llm(prompt)
        self.log(f"看空辩词: {thesis}")
        return {"agent": self.name, "strength": round(strength, 2), "thesis": thesis}

    def cross_examine(self, my_case: dict, opponent_case: dict) -> dict:
        self.log(f"⚔️ 收到多方阵营的炮火，准备发起反击...")
        if opponent_case["strength"] == 0.0:
            return my_case # 多方无话可说，无需反驳
            
        prompt = self.config.get(
            "cross_exam_template",
            "你代表【空方阵营】参与量化多空辩论。你的初始阵地与逻辑是：\n{my_case}\n现在，【多方阵营】对市场提出了以下看多逻辑和乐观预期：\n{opponent_case}\n作为看空辩手，请你对多方的盲目乐观进行一次冷酷、犀利的反驳(Cross-Examination)。如果多方理由十分坚固，也要适当收敛极端的做空态度。请将【你的原逻辑】与【对多方的反驳】融合为一篇全新的、更加具有警示性的空头最终陈词 (字数严格要求在150字以内)。"
        ).format(my_case=my_case['thesis'], opponent_case=opponent_case['thesis'])
        new_thesis = self.query_llm(prompt)
        self.log(f"看空反击陈词: {new_thesis[:50]}...")
        return {"agent": self.name, "strength": my_case["strength"], "thesis": new_thesis}

class GameReferee(BaseAgent):
    MIN_MEMORY_EVIDENCE = 3
    BULL_ADVANTAGE_THRESHOLD = 0.3
    MIN_TREND_STRENGTH_FOR_LIGHT_BUY = 0.12
    LIGHT_BUY_CONFIDENCE_FLOOR = 0.55

    def __init__(self, name: str, memory_bank=None):
        super().__init__(name, "多空博弈裁判")
        self.memory_bank = memory_bank
        self.config = ROLES_CONFIG.get("GameReferee", {})

    def _case_scores(self, cases: list[dict]) -> tuple[float, float]:
        bull_score = sum(float(c.get("bull_ratio", 0.5) or 0.5) * float(c.get("confidence", 0.5) or 0.5) for c in cases)
        bear_score = sum(float(c.get("bear_ratio", 0.5) or 0.5) * float(c.get("confidence", 0.5) or 0.5) for c in cases)
        return round(bull_score, 3), round(bear_score, 3)

    def _decision_from_cases(self, cases: list[dict]) -> tuple[str, float, float]:
        bull_score, bear_score = self._case_scores(cases)
        if bull_score > bear_score * 1.05:
            return "BUY", bull_score, bear_score
        if bear_score > bull_score * 1.05:
            return "SELL", bull_score, bear_score
        return "HOLD", bull_score, bear_score

    def _persist_human_feedback(self, ticker: str, human_comment: str, human_decision: str | None):
        if not self.memory_bank or not human_comment:
            return
        try:
            if hasattr(self.memory_bank, "append_experience"):
                try:
                    self.memory_bank.append_experience(
                        ticker=ticker,
                        role="Human",
                        sentiment_or_regime="ManualReview",
                        content=f"人工评论/修正沉淀: {human_comment}",
                        action_taken=human_decision or "COMMENT",
                        reward_score=0.0,
                    )
                except TypeError:
                    self.memory_bank.append_experience(
                        role="Human",
                        sentiment_or_regime="ManualReview",
                        content=f"人工评论/修正沉淀: {human_comment}",
                        action_taken=human_decision or "COMMENT",
                    )
            else:
                self.memory_bank.append({
                    "ticker": ticker,
                    "decision": human_decision or "HOLD",
                    "pnl_percent": 0.0,
                    "reward_score": 0.0,
                    "reflection_text": f"人工评论/修正沉淀: {human_comment}",
                    "math_stats": "人工经验沉淀，不作为收益样本。",
                })
            self.log("📝 已将人工评论/修正写入长期记忆。")
        except Exception as e:
            self.log(f"⚠️ 人工经验沉淀失败: {e}")

    def _judge_two_agent_disagreement(self, ticker: str, case_a: dict, case_b: dict, round_idx: int = 0) -> dict:
        prompt = (
            "你是裁判官。两个综合分析师对标的 {ticker} 方向不同，请判断这是【预期差买入机会】还是【一方明显错误】。\n"
            "分析师A：{agent_a}，方向={sent_a}，多={bull_a:.2f}，空={bear_a:.2f}，置信度={conf_a:.2f}，理由={reason_a}\n"
            "分析师B：{agent_b}，方向={sent_b}，多={bull_b:.2f}，空={bear_b:.2f}，置信度={conf_b:.2f}，理由={reason_b}\n"
            "当前轮次：{round_idx}\n"
            "只输出合法 JSON：{{\"mode\":\"expectation_gap/opponent_wrong/unclear\", "
            "\"decision\":\"BUY/SELL/HOLD\", \"confidence\":0.0到1.0, "
            "\"wrong_agent\":\"如果一方明显错误，填写其agent名，否则为空\", "
            "\"reasoning\":\"100字内裁判判断\", \"next_action\":\"下一步动作\"}}"
        ).format(
            ticker=ticker,
            agent_a=case_a.get("agent", "分析师A"),
            sent_a=case_a.get("sentiment", "neutral"),
            bull_a=case_a.get("bull_ratio", 0.5),
            bear_a=case_a.get("bear_ratio", 0.5),
            conf_a=case_a.get("confidence", 0.5),
            reason_a=case_a.get("reasoning", ""),
            agent_b=case_b.get("agent", "分析师B"),
            sent_b=case_b.get("sentiment", "neutral"),
            bull_b=case_b.get("bull_ratio", 0.5),
            bear_b=case_b.get("bear_ratio", 0.5),
            conf_b=case_b.get("confidence", 0.5),
            reason_b=case_b.get("reasoning", ""),
            round_idx=round_idx,
        )
        parsed = parse_llm_json(self.query_llm(prompt))
        decision = normalize_decision(parsed.get("decision"))
        if not decision:
            decision, _, _ = self._decision_from_cases([case_a, case_b])
        return {
            "mode": parsed.get("mode", "unclear"),
            "decision": decision,
            "confidence": float(parsed.get("confidence", 0.5) or 0.5),
            "wrong_agent": parsed.get("wrong_agent", ""),
            "reasoning": parsed.get("reasoning", "裁判未给出明确理由。"),
            "next_action": parsed.get("next_action", ""),
        }

    def step_agent_game(
        self,
        analyst_a,
        analyst_b,
        case_a: dict,
        case_b: dict,
        ticker: str = "UNKNOWN",
        max_depth: int | None = None,
        human_comment: str = "",
        human_decision: str | None = None,
    ):
        case_a = normalize_ratio_case(case_a.copy())
        case_b = normalize_ratio_case(case_b.copy())
        max_depth = int(max_depth if max_depth is not None else self.config.get("max_debate_depth", 2))
        debate_trace = []

        if case_a["sentiment"] == case_b["sentiment"]:
            decision, bull_score, bear_score = self._decision_from_cases([case_a, case_b])
            reasoning = f"两个综合分析师方向一致({case_a['sentiment']})，直接输出最终判断。"
            confidence = round((case_a["confidence"] + case_b["confidence"]) / 2, 3)
            mode = "same_direction"
        else:
            judge_view = self._judge_two_agent_disagreement(ticker, case_a, case_b, round_idx=0)
            debate_trace.append({"round": 0, "judge": judge_view, "case_a": case_a, "case_b": case_b})
            mode = judge_view.get("mode", "unclear")
            decision = judge_view["decision"]
            confidence = judge_view["confidence"]
            reasoning = judge_view["reasoning"]

            if mode != "expectation_gap":
                for round_idx in range(1, max_depth + 1):
                    case_a = analyst_a.debate_round(ticker, case_a, case_b, judge_view, round_idx)
                    case_b = analyst_b.debate_round(ticker, case_b, case_a, judge_view, round_idx)
                    judge_view = self._judge_two_agent_disagreement(ticker, case_a, case_b, round_idx=round_idx)
                    debate_trace.append({"round": round_idx, "judge": judge_view, "case_a": case_a, "case_b": case_b})

                    if case_a["sentiment"] == case_b["sentiment"]:
                        decision, _, _ = self._decision_from_cases([case_a, case_b])
                        confidence = round((case_a["confidence"] + case_b["confidence"]) / 2, 3)
                        reasoning = f"第 {round_idx} 轮后两个分析师观点达成一致：{case_a['sentiment']}。"
                        mode = "agent_consensus"
                        break
                    if judge_view.get("mode") == "expectation_gap":
                        decision = judge_view["decision"]
                        confidence = judge_view["confidence"]
                        reasoning = f"第 {round_idx} 轮裁判确认这是预期差机会：{judge_view['reasoning']}"
                        mode = "expectation_gap"
                        break
                else:
                    decision = judge_view["decision"]
                    confidence = judge_view["confidence"]
                    reasoning = f"达到博弈深度上限 {max_depth}，采用裁判最终判断：{judge_view['reasoning']}"
                    mode = "depth_limited_referee"

            bull_score, bear_score = self._case_scores([case_a, case_b])

        if human_decision:
            override = normalize_decision(human_decision)
            if override:
                decision = override
                reasoning += f" | 人工修正最终方向为 {override}。"
        self._persist_human_feedback(ticker, human_comment, human_decision)

        trend_strength = calculate_trend_strength(bull_score, bear_score)
        self.log(f"双分析师博弈裁决 -> 【{decision}】 mode={mode}, 深度={len(debate_trace)}")
        return {
            "decision": decision,
            "reason": reasoning,
            "bull_score": bull_score,
            "bear_score": bear_score,
            "confidence": float(confidence),
            "key_risks": [],
            "next_action": "按裁判裁决进入风控执行。",
            "trend_strength": trend_strength,
            "light_position": decision == "BUY" and mode == "expectation_gap",
            "debate_trace": debate_trace,
            "analyst_cases": [case_a, case_b],
            "human_comment": human_comment,
        }

    def step(self, bull_case, bear_case, ticker: str = "UNKNOWN", reports: list = None):
        self.log("⚖️ 正在进行高维裁判...")
        
        bull_thesis = bull_case["thesis"]
        bear_thesis = bear_case["thesis"]
        bull_score = bull_case["strength"]
        bear_score = bear_case["strength"]
        
        # 1. 触发 Memory RAG 检索！
        memory_prompt = ""
        current_scene_desc = f"我们在标的 {ticker} 上面临多方总票数 {bull_score:.2f} 对抗 空方总票数 {bear_score:.2f} 的战局。"
        if self.memory_bank:
            self.log("  -> 正在 RAG 检索历史上类似的多空战局踩坑经验...")
            past_exps = self.memory_bank.retrieve_relevant_experience(
                current_scene_desc=current_scene_desc, 
                role="System", 
                current_regime="General", 
                top_k=self.MIN_MEMORY_EVIDENCE
            )
            if len(past_exps) >= self.MIN_MEMORY_EVIDENCE:
                exp_strs = []
                for idx, exp in enumerate(past_exps):
                    exp_strs.append(f"   [历史战役 {idx+1}，经验置信度 {exp['score']:.2f}]: {exp['content']}")
                memory_prompt = "\n【来自系统 RAG 数据库的血泪教训警告】：\n" + "\n".join(exp_strs) + "\n请你在裁决时务必吸取上述历史错误或成功经验！如果历史表明极度危险，请果断修改你的决策或转为 HOLD。"
                self.log(f"  -> 检索到 {len(past_exps)} 条历史重要战役经验教训！")
            elif past_exps:
                self.log(f"  -> 仅检索到 {len(past_exps)} 条历史经验，样本不足 3 条，本轮不强行参考。")

        report_digest, weighted_bull_score, weighted_bear_score = build_report_digest(reports or [
            {"agent": "多方阵营", "sentiment": "positive", "confidence": bull_score, "reasoning": bull_thesis},
            {"agent": "空方阵营", "sentiment": "negative", "confidence": bear_score, "reasoning": bear_thesis},
        ])
        effective_bull_score = weighted_bull_score if reports else bull_score
        effective_bear_score = weighted_bear_score if reports else bear_score
        trend_strength = calculate_trend_strength(effective_bull_score, effective_bear_score)
        bull_has_clear_advantage = has_score_advantage(
            effective_bull_score,
            effective_bear_score,
            self.BULL_ADVANTAGE_THRESHOLD
        )

        prompt_template = self.config.get(
            "prompt_template",
            "你是一名理性的多空裁判。请综合多方、空方、历史记忆、趋势强度和证据摘要，输出 BUY/SELL/HOLD。\n证据摘要：\n{report_digest}\n历史记忆：\n{memory_prompt}\n趋势强度：{trend_strength:.2f}，越接近 1 越单边，越接近 0 越震荡。\n规则提示：如果多方得分比空方高出 30% 以上且趋势强度不弱，允许轻仓 BUY；历史记忆少于 3 条时不要强行据此转为 HOLD。\n输出纯 JSON：{{\"decision\": \"BUY/SELL/HOLD\", \"confidence\": 0.0到1.0的浮点数, \"reasoning\": \"100字内的裁决逻辑\", \"key_risks\": [\"风险1\", \"风险2\"], \"next_action\": \"下一步观察或执行建议\"}}"
        )
        
        prompt = prompt_template.format(
            bull_score=bull_score,
            bear_score=bear_score,
            bull_thesis=bull_thesis,
            bear_thesis=bear_thesis,
            report_digest=report_digest + f"\n\n[达利欧式可信度加权统计] 看多={weighted_bull_score:.2f} | 看空={weighted_bear_score:.2f}",
            memory_prompt=memory_prompt or "暂无足够历史记忆可用，本轮不以历史经验作为强制裁决依据。",
            trend_strength=trend_strength,
        )
        llm_result = self.query_llm(prompt)
        
        parsed = parse_llm_json(llm_result)
        decision = normalize_decision(parsed.get("decision") or parsed.get("sentiment") or parsed.get("结论"))
        if not decision:
            if "结论：BUY" in llm_result or "结论: BUY" in llm_result or "结论： BUY" in llm_result:
                decision = "BUY"
            elif "结论：SELL" in llm_result or "结论: SELL" in llm_result or "结论： SELL" in llm_result:
                decision = "SELL"
            elif "HOLD" in llm_result.upper() or "观望" in llm_result:
                decision = "HOLD"
            else:
                decision = "HOLD"

        confidence = parsed.get("confidence", 0.5)
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = 0.5

        reasoning = parsed.get("reasoning") or parsed.get("reason") or llm_result
        key_risks = parsed.get("key_risks", [])
        next_action = parsed.get("next_action", "")

        if (
            decision == "HOLD"
            and bull_has_clear_advantage
            and trend_strength >= self.MIN_TREND_STRENGTH_FOR_LIGHT_BUY
        ):
            decision = "BUY"
            confidence = max(float(confidence or 0.0), self.LIGHT_BUY_CONFIDENCE_FLOOR)
            reasoning = (
                f"{reasoning} | 多方得分较空方高出30%以上，趋势强度 {trend_strength:.2f}，"
                "按规则允许轻仓试探 BUY。"
            )
            next_action = next_action or "轻仓试探，后续观察趋势延续性与风险信号。"
            self.log("✅ 多方优势超过 30% 且趋势强度达标，将 HOLD 升级为轻仓 BUY。")
            
        self.log(f"裁判决断 -> 【{decision}】 深度理由:\n{llm_result}")
        
        # 将原始多空强度带入下一步计算仓位
        return {
            "decision": decision,
            "reason": reasoning,
            "bull_score": bull_score,
            "bear_score": bear_score,
            "confidence": float(confidence) if confidence is not None else 0.5,
            "key_risks": key_risks,
            "next_action": next_action,
            "trend_strength": trend_strength,
            "light_position": decision == "BUY" and bull_has_clear_advantage,
        }


# ==========================================
# 3. 执行与风控层 (Execution & Risk)
# ==========================================

class RiskManager(BaseAgent):
    MIN_MEMORY_EVIDENCE = 3
    LIGHT_POSITION_CAP = 0.15
    # A股默认长仓模式：SELL 仅表示看空/减仓，不执行做空仓位。
    ALLOW_SHORT_SELL = os.getenv("TRADING_ALLOW_SHORT", "0") == "1"

    def __init__(self, name: str, memory_bank):
        super().__init__(name, "首席风控官")
        self.memory_bank = memory_bank

    def step(self, ticker: str, referee_decision: dict):
        final_decision = referee_decision["decision"]
        reason = referee_decision["reason"]
        bull_score = referee_decision.get("bull_score", 0.0)
        bear_score = referee_decision.get("bear_score", 0.0)
        trend_strength = float(referee_decision.get("trend_strength", calculate_trend_strength(bull_score, bear_score)) or 0.0)
        is_light_position = bool(referee_decision.get("light_position", False))
        
        self.log("🛡️ 进行交易前的最后风控审查与动态仓位计算 (Kelly Criterion)...")
        kelly_fraction = 0.0

        if final_decision == "SELL" and not self.ALLOW_SHORT_SELL:
            final_decision = "HOLD"
            reason += " | A股长仓模式不执行做空，SELL 信号转为 HOLD（可理解为减仓/回避）。"
            self.log("⚠️ 长仓模式已启用，SELL 转为 HOLD，不分配做空仓位。")
        
        if self.memory_bank:
            # 【进阶方向1&5】：从高维向量数据库中调取高度相似的历史风险场景（取代生硬的时间倒推）
            try:
                scene_desc = f"当前计划对 {ticker} 执行 {final_decision}，裁判理由: {reason}"
                similar_experiences = self.memory_bank.retrieve_relevant_experience(
                    scene_desc,
                    role="System",
                    current_regime="General",
                    top_k=self.MIN_MEMORY_EVIDENCE
                )
                
                if len(similar_experiences) >= self.MIN_MEMORY_EVIDENCE:
                    self.log(f"🧠 联想到了 {len(similar_experiences)} 条相关的历史深刻教训...")
                    # 假如有极低分（亏损严重）的高相似度经验警告了该操作：
                    for exp in similar_experiences:
                        if "重创" in exp['content'] or "严重回撤" in exp['content']:
                            if final_decision != "HOLD":
                                final_decision = "HOLD"
                                reason += f" | 向量风控阻断: 提取到惨痛历史类似教训警示: {exp['content'][:50]}..."
                                self.log("⚠️ 触发[高维向量相似度检索]熔断，否决当前高危提议！")
                                break
                elif similar_experiences:
                    self.log(f"🧠 仅联想到 {len(similar_experiences)} 条相似经验，样本不足 3 条，不触发向量熔断。")
            except AttributeError:
                pass
                
            past_memories = self.memory_bank.get_recent_reflections(k=3, ticker=ticker)
            if len(past_memories) >= self.MIN_MEMORY_EVIDENCE:
                has_large_drawdown = any(mem.get('pnl_percent', 0) < -2.0 for mem in past_memories if mem.get('decision') in ["BUY", "SELL"])
                if has_large_drawdown and final_decision == "BUY":
                    final_decision = "HOLD"
                    reason += " | 基础风控阻断: 历史近期存在 >2% 重大回撤，强制熔断转为 HOLD！"
                    self.log("⚠️ 触发近期基础记忆风控熔断，否决买入提议！")
                else:
                    self.log("✅ 基础风控审核通过，近期无重大回撤隐患。")
            elif past_memories:
                self.log(f"✅ 近期基础记忆仅 {len(past_memories)} 条，样本不足 3 条，不触发强制熔断。")
                    
            # 动态计算凯利仓位
            if final_decision in ["BUY", "SELL"]:
                all_records = [m for m in self.memory_bank.memory if m.get('ticker') == ticker and m.get('decision') == final_decision]
                historical_pnls = [m.get('pnl_percent', 0.0) for m in all_records]
                
                N = len(historical_pnls)
                base_position = 0.0
                
                # 基于当期多空强度的初始建仓意愿
                total_score = bull_score + bear_score
                if total_score > 0:
                    if final_decision == "BUY":
                        base_position = min(bull_score / max(total_score, 1.0), 1.0)
                    elif final_decision == "SELL":
                        base_position = min(bear_score / max(total_score, 1.0), 1.0)
                
                # 如果有历史交易记录，则结合胜率和赔率计算 Kelly
                if N > 0:
                    win_pnls = [p for p in historical_pnls if p > 0]
                    loss_pnls = [p for p in historical_pnls if p < 0]
                    p_win = len(win_pnls) / N
                    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.001
                    avg_loss = abs(sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0.001
                    pnl_ratio = avg_win / avg_loss
                    
                    k_fraction = p_win - ((1 - p_win) / pnl_ratio)
                    # 只有大于0才认为有下注价值
                    kelly_weight = max(0.0, min(1.0, k_fraction))
                    
                    # 综合多空强度和凯利公式
                    final_position = (base_position * 0.4) + (kelly_weight * 0.6)
                    self.log(f"📊 基于历史 {N} 笔交易(胜率 {p_win*100:.1f}%)的 Kelly计算上限: {kelly_weight*100:.1f}%")
                else:
                    # 没有历史记录，保守建仓
                    final_position = base_position * 0.3
                    self.log(f"📊 无历史交易记录，保守计算置信仓位。")

                # 趋势越明确，越允许执行；趋势弱时维持轻仓，避免无方向震荡里重仓试错。
                trend_multiplier = 0.5 + min(1.0, trend_strength)
                final_position *= trend_multiplier
                if is_light_position:
                    final_position = min(final_position, self.LIGHT_POSITION_CAP)
                    self.log(f"🪶 裁判标记为轻仓试探，仓位上限限制为 {self.LIGHT_POSITION_CAP*100:.0f}%。")

                final_position = max(0.05, min(1.0, final_position)) # 最低5%试错，最高满仓
                kelly_fraction = round(final_position * 100, 2)
                reason += f" | 建议配置仓位: {kelly_fraction}%"
                self.log(f"💰 风控中心综合计算得出建议执行仓位: {kelly_fraction}%")

        # 封装成为最终交易对象的指令
        return {
            "decision": final_decision, 
            "reason": reason, 
            "position_percent": kelly_fraction
        }

class TraderAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, "交易执行机器人")
        self.config = ROLES_CONFIG.get("TraderAgent", {})

    def step(self, final_instruction: dict) -> str:
        decision = final_instruction["decision"]
        position = final_instruction.get("position_percent", 0.0)
        
        target_action = decision
        if decision in ["BUY", "SELL"] and position > 0:
            target_action = f"{decision} {position}%"
            
        execution_note = self.config.get("execution_note", "")
        self.log(f"💰 接收到上游最终指令: {target_action}。{execution_note} 向券商柜台/撮合引擎发送订单。")
        return target_action

class QuantitativeRiskReflector(BaseAgent):
    
    # 核心超参数配置：隔离硬编码的反馈调整阈值结构
    PARAMS = {
        "SEVERE_DRAWDOWN": -1.5,     # 严重回撤判定阈值 (%)
        "RECENT_PRINCIPLES": 10,     # 给LLM参考的最近公理数
        "DUPLICATE_CHECK_COUNT": 5,  # 查重的最近公理数
        "DUPLICATE_CHAR_THRES": 10   # 查重字符截断重合度阈值
    }

    def __init__(self, name: str, memory_bank):
        super().__init__(name, "量化分析与策略反思官")
        self.memory_bank = memory_bank
        self.config = ROLES_CONFIG.get("QuantitativeRiskReflector", {})
        
    def step(self, ticker: str, decision: str, reports: list, pnl_percent: float):
        # 提取真实意图与仓位（可能是 BUY 30.5%）
        action = "HOLD"
        position = 0.0
        if decision.startswith("BUY"):
            action = "BUY"
            try:
                # e.g., "BUY 27.5%" => extract 27.5
                position = float(decision.replace("BUY", "").replace("%", "").strip()) / 100.0
            except:
                position = 1.0
        elif decision.startswith("SELL"):
            action = "SELL"
            try:
                position = float(decision.replace("SELL", "").replace("%", "").strip()) / 100.0
            except:
                position = 1.0

        # 由于启用了动态仓位，真实的盈亏应当是：标的波动率 * 仓位暴露
        actual_pnl = pnl_percent * position if action == "BUY" else ((-pnl_percent) * position if action == "SELL" else 0.0)

        self.log(f"T+1日模拟结算复盘，标的[{ticker}] 原波动 [{pnl_percent}%], 实际账户盈亏贡献 [{actual_pnl:.3f}%]")
        
        ticker_records = [m for m in self.memory_bank.memory if m.get('ticker') == ticker]
        recent_actions = [m.get('decision', 'HOLD') for m in ticker_records]
        all_records = [m for m in ticker_records if m.get('decision') in ["BUY", "SELL"]]
        
        # 历史样本仅包含过去交易，避免在 reward 中重复把“当前样本”计入回撤惩罚。
        historical_pnls = [m.get('pnl_percent', 0.0) for m in all_records]
        stats_pnls = list(historical_pnls)
        if action in ["BUY", "SELL"]:
            stats_pnls.append(actual_pnl)
            
        N = len(stats_pnls)
        if N > 0:
            win_pnls = [p for p in stats_pnls if p > 0]
            loss_pnls = [p for p in stats_pnls if p < 0]
            
            p_win = len(win_pnls) / N
            avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
            avg_loss = abs(sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0.0
            pnl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            if pnl_ratio != float('inf') and pnl_ratio > 0:
                kelly_fraction = p_win - ((1 - p_win) / pnl_ratio)
            else:
                kelly_fraction = p_win
                
            kelly_fraction = max(0.0, min(1.0, kelly_fraction)) * 100
            expected_return = p_win * avg_win - (1 - p_win) * avg_loss
            
            stats_msg = f"样本={N}笔, 胜率={p_win*100:.1f}%, 真实盈亏比={pnl_ratio if pnl_ratio != float('inf') else 999.99:.2f}, 期望单次收益={expected_return:.2f}%. Kelly建议上限: {kelly_fraction:.1f}%"
        else:
            stats_msg = "暂无足够实盘买卖样本计算凯利仓位与胜率。"

        reward = compute_trade_reward(
            action=action,
            actual_pnl_percent=actual_pnl,
            market_move_percent=pnl_percent,
            # 只传入“过去样本”，当前样本由 compute_trade_reward 内部统一拼接处理。
            historical_pnls=historical_pnls,
            recent_actions=recent_actions,
            position=position,
        )

        reward_msg = (
            f"reward={reward.reward:.3f}, return={reward.return_component:.3f}, "
            f"risk_penalty={reward.volatility_penalty + reward.drawdown_penalty + reward.loss_streak_penalty:.3f}, "
            f"behavior_penalty={reward.turnover_penalty + reward.exposure_penalty + reward.hold_penalty + reward.hold_streak_penalty:.3f}"
        )

        reflection_text = ""
        if action in ["BUY", "SELL"]:
            if actual_pnl < self.PARAMS.get("SEVERE_DRAWDOWN", -1.5):
                reflection_text = f"严重回撤 ({actual_pnl:.2f}%)！当前统计: {stats_msg}。系统应考虑降低贝塔参与度。"
            elif actual_pnl > 0:
                reflection_text = f"策略获利 ({actual_pnl:.2f}%)。当前统计: {stats_msg}。"
            else:
                reflection_text = f"微小摩擦 ({actual_pnl:.2f}%)。当前统计: {stats_msg}"
        else:
            reflection_text = f"本次为空仓观望(HOLD)，连续观望 {reward.recent_hold_streak} 次，已计入机会成本惩罚。"
            
        record = {
            "ticker": ticker,
            "decision": action,
            "pnl_percent": round(actual_pnl, 2),
            "market_move_percent": round(pnl_percent, 2),
            "position": round(position, 4),
            "reward_score": round(reward.reward, 4),
            "reward_text": reward_msg,
            "reward_stats": reward.to_dict(),
            "reflection_text": reflection_text,
            "math_stats": stats_msg
        }
        
        # 将新经验记录，并且触发RAG后台自动向量化并打标签
        self.memory_bank.append(record)
        
        # 【进阶方向4】：经验反馈形成强化学习闭环
        if action in ["BUY", "SELL"] and actual_pnl != 0:
            self.memory_bank.update_experience_score_by_action(
                ticker=ticker, 
                action_taken=action, 
                reward_signal=reward.reward
            )
            
        # 【进阶方向3】：偶尔尝试触发结晶
        try:
            high_value_materials = self.memory_bank.crystallize_knowledge(None)
            if high_value_materials:
                self.log(f"🧠 [自我进化] 检测到了高分致胜经验池，正在呼叫大模型将其结晶为公理法则！")
                
                # 读取已有原则，防止大模型复读
                existing_principles = "无"
                if len(self.memory_bank.principles) > 0:
                    # 取最近的 N 条参考值
                    recent_n = self.PARAMS.get("RECENT_PRINCIPLES", 10)
                    recent = [p.get("principle", "") for p in self.memory_bank.principles[-recent_n:]]
                    existing_principles = "\n- " + "\n- ".join(recent)
                
                prompt = self.config.get(
                    "principle_prompt_template",
                    "你是一名为对冲基金撰写《内部交易原则》(Redbook)的风控总监。以下是系统近期积累的实盘交易记忆（包含高价值致胜经验与亏损的证伪反例）：\n{high_value_materials}\n\n已有公理参考：\n{existing_principles}\n\n请在提炼时严格遵循【反例证伪（Falsifiability）机制】：\n1. 任何原则都必须带有边界和适用条件，请结合提供的[失败教训]来界定这条新经验的“死穴”或“失效场景”。\n2. 仔细对比已有公理，寻找反直觉角度，严禁重复。\n\n请提炼为1条【具有普适性且带有证伪条件的量化交易公理】(格式：核心法则 + 失效边界)。不能超过60个字，要求极度精炼。"
                ).format(high_value_materials=high_value_materials, existing_principles=existing_principles)
                
                principle_text = self.query_llm(prompt)
                
                # 提取配置参数
                dup_count = self.PARAMS.get("DUPLICATE_CHECK_COUNT", 5)
                char_thres = self.PARAMS.get("DUPLICATE_CHAR_THRES", 10)
                
                # 再次过滤，如果返回的内容完全雷同，则抛弃
                if len(self.memory_bank.principles) > 0 and principle_text[:char_thres] in "".join([p.get("principle", "") for p in self.memory_bank.principles[-dup_count:]]):
                    self.log(f"⚠️ [复读机拦截] 本次结晶的知识过度雷同，已放弃入库。")
                else:
                    # 存入到原则文件中，并更新向量库标记（略去复杂标记以防覆盖，只存json）
                    import datetime
                    self.memory_bank.principles.append({
                        "date": datetime.datetime.now().strftime('%Y-%m-%d'),
                        "ticker": ticker,
                        "principle": principle_text
                    })
                    self.memory_bank._save_principles()
                    self.log(f"📜 [全新底层原则已确立并入库]: {principle_text}")
        except Exception as e:
            pass

        self.log(f"【进化总结】: {reflection_text} | {reward_msg}")
        return reflection_text
