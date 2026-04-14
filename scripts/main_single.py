import os
import sys

# 添加PYTHONPATH，方便模块查找
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.roles import NewsAnalystAgent, QuantResearcherAgent, TraderAgent, QuantitativeRiskReflector
from memory.memory_bank import MemoryBank
from rag.retriever import SimpleRAG
from dl.predictor import DLEngine
import logging

def main():
    print("=== 初始化 A股智能投研多智能体系统 (Demo) ==================")
    print("本系统融合 [RAG向量知识库] 与 [受Pytorch/Tensorflow支持的DeepLearning模型]")
    print("新增功能：[ReAct 反思迭代记忆体]，实现模型自我闭环进化")
    print("============================================================\n")

    # 1. 载入全局长期记忆突触
    memory_bank = MemoryBank()

    # 1. 挂载基座引擎
    mock_news_kb = [
        "贵州茅台 (600519) 发布2026年三季报，利润大增",
        "五粮液近期受政策鼓励消费，股价反弹",
        "宁德时代获得百亿电池大单，新能源汽车板块活跃",
        "中芯国际突破关键光刻技术，半导体国产替代加速",
        "政策出台全面降低印花税，A股全线上涨"
    ]
    
    # 3. 开启投研作业流程
    ticker = "600519"
    print(f"\n>>>> 收到指令，开始对标的 [{ticker}] 进行多角度研判 <<<<")

    # 获取真实新闻数据替代模拟数据
    import akshare as ak
    import pandas as pd
    
    print("\n>>>> 正在通过 AKShare 获取最新的市场/个股真实新闻作 RAG 知识库... <<<<")
    try:
        # 使用 akshare 获取东方财富的实时财经新闻或者指定个股新闻
        news_df = ak.stock_news_em(symbol=ticker)
        # 提取前 10 条真实新闻的新闻标题及内容
        if not news_df.empty:
            real_news_kb = news_df['新闻标题'].head(10).tolist()
            print(f"成功获取到 {len(real_news_kb)} 条关于 [{ticker}] 的真实新闻: \n" + "\n".join([f"- {n}" for n in real_news_kb]))
        else:
            real_news_kb = ["未查询到相关新闻"]
    except Exception as e:
        print(f"真实新闻获取失败: {e}")
        real_news_kb = mock_news_kb

    rag_engine = SimpleRAG(data_sources=real_news_kb)
    dl_engine = DLEngine()
    
    # 2. 初始化智能体群组（扮演多头/空头辩论或研报写手）
    analyst_agent = NewsAnalystAgent(name="AI基本面研究员", rag_engine=rag_engine)
    quant_agent = QuantResearcherAgent(name="DeepSeek量化专家", dl_engine=dl_engine)
    trader_agent = TraderAgent(name="主理人/风控负责人", memory_bank=memory_bank)
    reflector_agent = QuantitativeRiskReflector(name="量化模型风控官", memory_bank=memory_bank)
    
    report_1 = analyst_agent.step(ticker)
    report_2 = quant_agent.step(ticker)
    
    # 专家小组进行结论汇总与决策博弈
    reports = [report_1, report_2]
    
    # 最后由主理人给出交易策略
    print("\n>>>> 主理人收集研报与经验记忆进行辩论与最终决断 <<<<")
    decision = trader_agent.step(ticker, reports)

    # 4. 前往时间切片，接入真实市场当期/次日行情，并由评估官复盘
    print("\n>>>> 进入真实时空盘后结算，获取市场真实盈亏并开启反思阶段 <<<<")
    try:
        import pandas as pd
        prefix = "sh" if ticker.startswith("6") else "sz"
        prefixed = f"{prefix}{ticker}"
        df_hist = ak.stock_zh_a_daily(symbol=prefixed, adjust="qfq")
        
        df_hist.rename(columns={
            'date': '日期', 'open': '开盘', 'high': '最高', 'low': '最低', 
            'close': '收盘', 'volume': '成交量', 'amount': '成交额', 'turnover': '换手率'
        }, inplace=True)
        # 获取衍生特征以满足DL模块要求
        df_hist['日期'] = pd.to_datetime(df_hist['日期']).dt.strftime('%Y-%m-%d')
        df_hist['前收盘'] = df_hist['收盘'].shift(1)
        df_hist['涨跌额'] = df_hist['收盘'] - df_hist['前收盘']
        df_hist['涨跌幅'] = df_hist['涨跌额'] / df_hist['前收盘'] * 100
        df_hist['振幅'] = (df_hist['最高'] - df_hist['最低']) / df_hist['前收盘'] * 100
        df_hist['换手率'] = df_hist['换手率'] * 100 # 转换为百分比
        df_hist.dropna(inplace=True)
        df_hist.reset_index(drop=True, inplace=True)
        if not df_hist.empty:
            latest_date = df_hist['日期'].iloc[-1]
            real_pnl = float(df_hist['涨跌幅'].iloc[-1])
            print(f"成功获取 [{ticker}] 最新交易日 ({latest_date}) 的真实收盘涨跌幅为: {real_pnl}%")
        else:
            real_pnl = 0.0
    except Exception as e:
        print(f"真实盈亏行情获取失败，重置为0: {e}")
        real_pnl = 0.0
    
    reflector_agent.step(ticker, decision, reports, pnl_percent=real_pnl)

    print("\n=== 行动执行完毕 ===")
    
if __name__ == "__main__":
    main()
