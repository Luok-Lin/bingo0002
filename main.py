import os
import sys
import urllib.request

# 强行拦截并绕过 macOS 系统的底层网络配置中的全局代理
# 防止 requests 库在使用 AKShare 时隐式读取局域网/Wi-Fi配置的梯子代理导致连接重置
urllib.request.getproxies = lambda: {}

# 临时清除所有代理环境变量，防止拉取境内金融API(东方财富/AKShare)被VPN梯子拒断
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加PYTHONPATH，方便模块查找
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.roles import (
    TechnicalAnalyst, SentimentAnalyst, FundamentalAnalyst,
    MacroAnalyst, SmartMoneyAnalyst, NewsAnalystAgent,
    QuantResearcherAgent, BullResearcher, BearResearcher,
    GameReferee, RiskManager, TraderAgent, QuantitativeRiskReflector
)
from memory.memory_bank import MemoryBank
from rag.retriever import SimpleRAG
from dl.predictor import DLEngine
import akshare as ak
import pandas as pd
import time
import datetime
import concurrent.futures

def main():
    print("=== 启动 A股智能投研多智能体系统：全年历史回测模式 ==================")
    print("本回测将跨越过去约3年（全真实历史数据）。")
    print("以天为单位滑动，收集每日特征 -> 做出交易决策 -> T+1日结算盈亏存入经验库。")
    print("========================================================================\n")

    # 1. 载入全局长期记忆突触
    memory_bank = MemoryBank()

    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        backtest_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    else:
        try:
            ticker = ""
            while not ticker:
                ticker = input("请输入您要分析的 A股 股票代码 (例如: 600519，000001 等): ").strip()
                if not ticker:
                    print("股票代码不能为空，请重新输入。")
            
            backtest_days_input = input("请输入您想回测的最近交易日天数 (直接回车默认一整年): ").strip()
            if backtest_days_input and backtest_days_input.isdigit():
                backtest_limit = int(backtest_days_input)
            else:
                backtest_limit = None
        except (EOFError, KeyboardInterrupt):
            print("\n程序已取消。")
            return
            
    print(f"\n>>>> 正在从 AKShare 拉取 [{ticker}] 的过去一整年K线数据进行切片循环 ... <<<<")
    
    # 获取接近 1 年的日线行情数据（以 250 个交易日约1年计，我们将拉去稍多一些方便计算初始特征）
    try:
        # 获取最近 300 个交易日以涵盖整个前一年并留有10天特征冗余
        df_hist = ak.stock_zh_a_hist(symbol=ticker, period="daily", adjust="qfq")
        # 修改为获取过去 3 年的数据 (大约 750 个交易日)
        df_hist = df_hist.tail(750).reset_index(drop=True)
        if df_hist.empty or len(df_hist) < 20:
            print("历史数据拉取失败或不足。退出。")
            return
        print(f"成功获取 [{ticker}] 的三年级别历史行情，共计 {len(df_hist)} 个交易日！")
    except Exception as e:
        print(f"历史数据获取致命错误: {e}")
        return

    # 通过 API 获取标的真实历史新闻，代替 mock
    print(f"\n[系统新闻获取] 获取真实的 {ticker} 新鲜历史新闻以构建知识库 (RAG DB) ...")
    real_news_kb = []
    
    # 1. 抓取新闻
    try:
        news_df = ak.stock_news_em(symbol=ticker)
        for _, row in news_df.head(100).iterrows():
            pub_time = str(row.get('发布时间', '2000-01-01'))
            try:
                date_int = int(pub_time[:10].replace('-', ''))
            except:
                date_int = 20000101
                
            real_news_kb.append({
                "page_content": "[外围资讯] " + str(row['新闻标题']) + " : " + str(row['新闻内容']),
                "metadata": {"date_int": date_int}
            })
        print(f"成功攫取 {len(news_df)} 条关于 {ticker} 的近期新闻。")
    except Exception as e:
        print(f"新闻抓取受限: {e}")

    # 2. 抓取机构历史深度研报 (大幅扩充RAG库容量与深度)
    print(f"[研报与深度评级] 正在获取 {ticker} 历史券商研报补充认知...")
    try:
        report_df = ak.stock_research_report_em(symbol=ticker)
        if report_df is not None and not report_df.empty:
            for _, row in report_df.iterrows():
                pub_time = str(row.get('日期', '2000-01-01'))
                try:
                    date_int = int(pub_time[:10].replace('-', ''))
                except:
                    date_int = 20000101
                
                content = f"[券商研报] 机构: {row.get('机构', '未知')} | 评级: {row.get('东财评级', '未知')} | 核心观点摘要: {row.get('报告名称', '')}"
                real_news_kb.append({
                    "page_content": content,
                    "metadata": {"date_int": date_int}
                })
            print(f"成功攫取 {len(report_df)} 份专业研报注入向知识库。")
    except Exception as e:
        print(f"研报提取遭遇阻碍: {e}")

    if not real_news_kb:
        print("未抓取到任何外部文本，使用内置降级备用认知。")
        real_news_kb = [
            {"page_content": "贵州茅台发布财报，利润大增", "metadata": {"date_int": 20240101}},
            {"page_content": "白酒消费回暖", "metadata": {"date_int": 20240101}},
            {"page_content": "宏观经济继续修复", "metadata": {"date_int": 20240101}},
            {"page_content": "市场出现资金净流入", "metadata": {"date_int": 20240101}}
        ]
        
    print(f"✅ RAG 混合语料筹备完毕，总计向 ChromaDB 灌入 {len(real_news_kb)} 条高维投研文本。")

    rag_engine = SimpleRAG(data_sources=real_news_kb)
    dl_engine = DLEngine()

    # == 使用回测的前60天作为真实的强化学习数据，真实训练量化模型 ==
    print("\n>>>> 用前置的 60 天真实盘面数据训练 LSTM 量化预测模型 <<<<")
    train_df = df_hist.head(60) # 截取前60天数据训练
    dl_engine.train_on_history(train_df, epochs=20)
    
    # == 实例化全链路多智能体团队 ==
    # 1. 分析团队
    tech_analyst = TechnicalAnalyst(name="技术面分析师")
    sentiment_analyst = SentimentAnalyst(name="舆情分析师")
    fund_analyst = FundamentalAnalyst(name="基本面分析师")
    macro_analyst = MacroAnalyst(name="宏观分析师")
    smart_analyst = SmartMoneyAnalyst(name="主力资金分析师")
    news_analyst = NewsAnalystAgent(name="新闻研报专家", rag_engine=rag_engine)
    quant_agent = QuantResearcherAgent(name="深度学习量化专家", dl_engine=dl_engine)
    
    # 2. 博弈层与执行层
    bull_researcher = BullResearcher(name="看多金牌辩手")
    bear_researcher = BearResearcher(name="看空金牌辩手")
    referee = GameReferee(name="无情裁判官", memory_bank=memory_bank)
    risk_manager = RiskManager(name="风控大脑", memory_bank=memory_bank)
    trader_agent = TraderAgent(name="极速交易接口")
    reflector_agent = QuantitativeRiskReflector(name="量化策略迭代官", memory_bank=memory_bank)
    
    start_index = 60 # 从已经过拟合/训练完毕后的第60天起开始向后滚动盘面
    total_days = len(df_hist) - 1 # 留最后一天给 T+1 用

    if backtest_limit is not None and backtest_limit > 0:
        # 如果用户指定了回测天数，我们把 start_index 往后推，只跑最近的 N 天
        override_start = total_days - backtest_limit
        start_index = max(start_index, override_start)
        print(f"\n[测试模式] 已激活: 仅回测最近的 {backtest_limit} 个交易日...")
    
    for i in range(start_index, total_days):
        target_date = df_hist.iloc[i]['日期']
        print(f"\n======== 【时间游标滑动】当前日期: {target_date} ========")
        
        # 截取 T 日及过往 10 天的数据作为量化模型的特征张量矩阵
        window_df = df_hist.iloc[i-10 : i]
        features = window_df[['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']].values
        
        # 阶段1: 并行调研产出
        print("  [System] 启动底层多智能体兵团并发执行 (Async Threads)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            future_tech = executor.submit(tech_analyst.step, ticker, features=features, target_date=str(target_date))
            future_sent = executor.submit(sentiment_analyst.step, ticker, target_date=str(target_date))
            future_fund = executor.submit(fund_analyst.step, ticker, target_date=str(target_date))
            future_macro = executor.submit(macro_analyst.step, ticker, target_date=str(target_date))
            future_smart = executor.submit(smart_analyst.step, ticker, target_date=str(target_date))
            future_news = executor.submit(news_analyst.step, ticker, target_date=str(target_date))
            future_quant = executor.submit(quant_agent.step, ticker, features_override=features, target_date=str(target_date))

            r_tech = future_tech.result()
            r_sent = future_sent.result()
            r_fund = future_fund.result()
            r_macro = future_macro.result()
            r_smart = future_smart.result()
            r_news = future_news.result()
            r_quant = future_quant.result()
        print("  [System] 并发调研阶段 1 完成，收集汇报完毕。")
        
        all_reports = [r_tech, r_sent, r_fund, r_macro, r_smart, r_news, r_quant]
        
        # 阶段2: 多空博弈与裁决
        bull_initial = bull_researcher.step(all_reports)
        bear_initial = bear_researcher.step(all_reports)
        
        # 交叉质询辩论轮次 (Cross-Examination)
        bull_case = bull_researcher.cross_examine(my_case=bull_initial, opponent_case=bear_initial)
        bear_case = bear_researcher.cross_examine(my_case=bear_initial, opponent_case=bull_initial)
        
        # 裁判最终定夺
        referee_decision = referee.step(bull_case, bear_case, ticker=ticker, reports=all_reports)
        
        # 阶段3: 风控与执行
        final_instruction = risk_manager.step(ticker, referee_decision)
        decision = trader_agent.step(final_instruction)

        # 阶段4: 模拟 T+1 日，去取第二天（i+1 行）的真实收盘涨跌幅！
        next_day_date = df_hist.iloc[i+1]['日期']
        real_pnl = float(df_hist.iloc[i+1]['涨跌幅'])
        
        print(f"  [游标揭晓] 进入 T+1日 ({next_day_date})，市场真实收盘涨跌幅为: {real_pnl}%")
        
        # 迭代官完成记录与数学期望计算
        reflector_agent.step(ticker, decision, all_reports, pnl_percent=real_pnl)
        
        # 为了不刷屏太快，微微暂停 0.1秒
        time.sleep(0.1)
   
if __name__ == "__main__":
    main()
