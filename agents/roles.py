from .base import BaseAgent
import numpy as np
import time
from dataflows.providers.akshare_provider import AkShareProvider
import json
import os

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
    try:
        # 清理可能包含的markdown json代码块标记
        cleaned = llm_result.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except Exception as e:
        # Fallback 策略
        sentiment = "neutral"
        if "positive" in llm_result.lower(): sentiment = "positive"
        elif "negative" in llm_result.lower(): sentiment = "negative"
        return {"sentiment": sentiment, "reasoning": llm_result, "thought_process": "解析失败，无法获取思维链", "confidence": 0.5}


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
        news_str = provider.fetch_sentiment_data(ticker)
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
        data_str = provider.fetch_fundamental_data(ticker)
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
        macro_str = provider.fetch_macro_data()
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
        flow_str = provider.fetch_smart_money_data(ticker)
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
        
        max_iters = 5
        query = ticker
        all_docs = []
        
        # Agentic RAG 反思检索循环
        for i in range(max_iters):
            self.log(f"  -> 第 {i+1} 轮检索, 搜索词: '{query}'")
            docs = self.rag.retrieve(query, target_date=target_date, top_k=2)
            if docs:
                all_docs.extend(docs)
                
            # 去重
            all_docs = list(set(all_docs))
            if not all_docs:
                self.log("  -> 未检索到任何有效外部文本。结束检索。")
                break
                
            summary = "\n".join(all_docs)
            eval_prompt = self.config.get(
                "eval_prompt_template",
                "你是一个智能搜索研报与新闻的特工。我们正在分析股票 【{ticker}】。目前我们收集到的情报如下：\n{summary}\n请你判断这些信息是否足够支撑判断该股票的涨跌趋势？请务必返回合法 JSON：{{\"enough\": true或false, \"next_query\": \"如果不够请给更有针对性的搜索词\", \"reason\": \"50字内理由\"}}"
            ).format(ticker=ticker, summary=summary)
            
            try:
                eval_result = self.query_llm(eval_prompt)
                parsed_eval = parse_llm_json(eval_result)
                is_enough = parsed_eval.get("enough", True)
                reason = parsed_eval.get("reason", "无法解析评估理由。")
                
                self.log(f"  -> RAG 评估结果: 足够={is_enough}, 理由: {reason}")
                
                if is_enough:
                    break
                else:
                    query = parsed_eval.get("next_query", ticker + " 产业链 财务")
                    if not query or query == ticker:
                        break
            except Exception as e:
                self.log(f"  -> Agentic 评估环节出错，中断检索循环: {e}")
                break

        if not all_docs:
            all_docs = ["暂无有关此标的的新闻研报或宏观有效资讯记录。"]
            
        final_summary = "总体文献摘要：\n" + "\n".join(all_docs)
        self.scratchpad.append(f"Agentic RAG 工作流结束。收集到文献数为: {len(all_docs)}")
        
        # 最终汇总判定
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
        
        self.log(f"LSTM 预测 (无情绪纯数学打分): {pred['trend']} 原始分[{dl_score:.4f}] 预测置信度[{dl_confidence:.2f}]")
        
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
    def __init__(self, name: str, memory_bank=None):
        super().__init__(name, "多空博弈裁判")
        self.memory_bank = memory_bank
        self.config = ROLES_CONFIG.get("GameReferee", {})

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
                top_k=2
            )
            if past_exps:
                exp_strs = []
                for idx, exp in enumerate(past_exps):
                    exp_strs.append(f"   [历史战役 {idx+1}，经验置信度 {exp['score']:.2f}]: {exp['content']}")
                memory_prompt = "\n【来自系统 RAG 数据库的血泪教训警告】：\n" + "\n".join(exp_strs) + "\n请你在裁决时务必吸取上述历史错误或成功经验！如果历史表明极度危险，请果断修改你的决策或转为 HOLD。"
                self.log(f"  -> 检索到 {len(past_exps)} 条历史重要战役经验教训！")

        report_digest, weighted_bull_score, weighted_bear_score = build_report_digest(reports or [
            {"agent": "多方阵营", "sentiment": "positive", "confidence": bull_score, "reasoning": bull_thesis},
            {"agent": "空方阵营", "sentiment": "negative", "confidence": bear_score, "reasoning": bear_thesis},
        ])

        prompt_template = self.config.get(
            "prompt_template",
            "你是一名理性的多空裁判。请综合多方、空方、历史记忆和证据摘要，输出 BUY/SELL/HOLD。\n证据摘要：\n{report_digest}\n历史记忆：\n{memory_prompt}\n输出纯 JSON：{{\"decision\": \"BUY/SELL/HOLD\", \"confidence\": 0.0到1.0的浮点数, \"reasoning\": \"100字内的裁决逻辑\", \"key_risks\": [\"风险1\", \"风险2\"], \"next_action\": \"下一步观察或执行建议\"}}"
        )
        
        prompt = prompt_template.format(
            bull_score=bull_score,
            bear_score=bear_score,
            bull_thesis=bull_thesis,
            bear_thesis=bear_thesis,
            report_digest=report_digest + f"\n\n[达利欧式可信度加权统计] 看多={weighted_bull_score:.2f} | 看空={weighted_bear_score:.2f}",
            memory_prompt=memory_prompt or "暂无历史记忆可用。"
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
        }


# ==========================================
# 3. 执行与风控层 (Execution & Risk)
# ==========================================

class RiskManager(BaseAgent):
    def __init__(self, name: str, memory_bank):
        super().__init__(name, "首席风控官")
        self.memory_bank = memory_bank

    def step(self, ticker: str, referee_decision: dict):
        final_decision = referee_decision["decision"]
        reason = referee_decision["reason"]
        bull_score = referee_decision.get("bull_score", 0.0)
        bear_score = referee_decision.get("bear_score", 0.0)
        
        self.log("🛡️ 进行交易前的最后风控审查与动态仓位计算 (Kelly Criterion)...")
        kelly_fraction = 0.0
        
        if self.memory_bank:
            # 【进阶方向1&5】：从高维向量数据库中调取高度相似的历史风险场景（取代生硬的时间倒推）
            try:
                scene_desc = f"当前计划对 {ticker} 执行 {final_decision}，裁判理由: {reason}"
                similar_experiences = self.memory_bank.retrieve_relevant_experience(scene_desc, role="System", current_regime="General", top_k=2)
                
                if similar_experiences:
                    self.log(f"🧠 联想到了 {len(similar_experiences)} 条相关的历史深刻教训...")
                    # 假如有极低分（亏损严重）的高相似度经验警告了该操作：
                    for exp in similar_experiences:
                        if "重创" in exp['content'] or "严重回撤" in exp['content']:
                            if final_decision != "HOLD":
                                final_decision = "HOLD"
                                reason += f" | 向量风控阻断: 提取到惨痛历史类似教训警示: {exp['content'][:50]}..."
                                self.log("⚠️ 触发[高维向量相似度检索]熔断，否决当前高危提议！")
                                break
            except AttributeError:
                pass
                
            past_memories = self.memory_bank.get_recent_reflections(k=3, ticker=ticker)
            if past_memories:
                has_large_drawdown = any(mem.get('pnl_percent', 0) < -2.0 for mem in past_memories if mem.get('decision') in ["BUY", "SELL"])
                if has_large_drawdown and final_decision == "BUY":
                    final_decision = "HOLD"
                    reason += " | 基础风控阻断: 历史近期存在 >2% 重大回撤，强制熔断转为 HOLD！"
                    self.log("⚠️ 触发近期基础记忆风控熔断，否决买入提议！")
                else:
                    self.log("✅ 基础风控审核通过，近期无重大回撤隐患。")
                    
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
        
        all_records = [m for m in self.memory_bank.memory if m.get('ticker') == ticker and m.get('decision') in ["BUY", "SELL"]]
        
        historical_pnls = [m.get('pnl_percent', 0.0) for m in all_records]
        if action in ["BUY", "SELL"]:
            historical_pnls.append(actual_pnl)
            
        N = len(historical_pnls)
        if N > 0:
            win_pnls = [p for p in historical_pnls if p > 0]
            loss_pnls = [p for p in historical_pnls if p < 0]
            
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

        reflection_text = ""
        if action in ["BUY", "SELL"]:
            if actual_pnl < -1.5:
                reflection_text = f"严重回撤 ({actual_pnl:.2f}%)！当前统计: {stats_msg}。系统应考虑降低贝塔参与度。"
            elif actual_pnl > 0:
                reflection_text = f"策略获利 ({actual_pnl:.2f}%)。当前统计: {stats_msg}。"
            else:
                reflection_text = f"微小摩擦 ({actual_pnl:.2f}%)。当前统计: {stats_msg}"
        else:
            reflection_text = "本次为空仓观望(HOLD)，未产生资金损患。"
            
        record = {
            "ticker": ticker,
            "decision": action,
            "pnl_percent": round(actual_pnl, 2),
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
                pnl_result=actual_pnl
            )
            
        # 【进阶方向3】：偶尔尝试触发结晶
        try:
            high_value_materials = self.memory_bank.crystallize_knowledge(None)
            if high_value_materials:
                self.log(f"🧠 [自我进化] 检测到了高分致胜经验池，正在呼叫大模型将其结晶为公理法则！")
                prompt = self.config.get(
                    "principle_prompt_template",
                    "你是一名为对冲基金撰写《内部交易原则》(Redbook)的总监。以下是系统在实盘后积累的、被验证赚了钱的优质高分经验片段：\n{high_value_materials}\n请你将这些碎片提炼为1条【高度抽象、具有普适性的量化交易公理】，不能超过40个字，要求极度精炼。"
                ).format(high_value_materials=high_value_materials)
                principle_text = self.query_llm(prompt)
                
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

        self.log(f"【进化总结】: {reflection_text}")
        return reflection_text
