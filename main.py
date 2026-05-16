import os
import argparse
import sys
import subprocess
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
    TechnicalFlowAnalyst, FundamentalNewsAnalyst,
    GameReferee, RiskManager, TraderAgent, QuantitativeRiskReflector
)
from memory.memory_bank import MemoryBank
from rag.retriever import SimpleRAG
import akshare as ak
import pandas as pd
import time
import datetime
import concurrent.futures
import json


def parse_args():
    parser = argparse.ArgumentParser(description="A股智能投研多智能体系统")
    parser.add_argument("ticker", nargs="?", default=None, help="A股股票代码，例如 600519")
    parser.add_argument("backtest_limit", nargs="?", type=int, default=None, help="最近回测天数")
    parser.add_argument("--auto-tune", action="store_true", help="回测结束后自动运行 reward 调参")
    parser.add_argument("--tune-window", type=int, default=120, help="自动调参时使用最近多少条回测记录")
    parser.add_argument("--tune-samples", type=int, default=80, help="自动调参时随机采样候选数量")
    parser.add_argument("--tune-grid", action="store_true", help="自动调参时使用网格搜索")
    parser.add_argument("--tune-dry-run", action="store_true", help="自动调参只生成报告，不写回配置")
    parser.add_argument("--tune-output-path", default=None, help="自动调参输出配置路径，默认覆盖 rl/reward_config.json")
    parser.add_argument("--tune-report-dir", default=None, help="自动调参图表报告目录")
    parser.add_argument("--no-train", action="store_true", help="跳过 DL 模型训练（用于加速回测）")
    parser.add_argument("--train-epochs", type=int, default=20, help="训练时使用的 epochs 数量（默认20）")
    parser.add_argument("--debate-depth", type=int, default=2, help="两个综合分析师与裁判官的最大博弈轮数")
    parser.add_argument("--human-comment", default="", help="人工评论，将作为经验沉淀写入记忆库")
    parser.add_argument("--human-decision", default=None, help="人工修正最终方向，可选 BUY/SELL/HOLD")
    return parser.parse_args()


def run_auto_tune(args):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "auto_tune_reward.py")
    cmd = [
        sys.executable,
        script_path,
        "--window", str(args.tune_window),
        "--samples", str(args.tune_samples),
    ]
    if args.tune_grid:
        cmd.append("--grid")
    if args.tune_dry_run:
        cmd.append("--dry-run")
    if args.tune_output_path:
        cmd.extend(["--output-path", args.tune_output_path])
    if args.tune_report_dir:
        cmd.extend(["--report-dir", args.tune_report_dir])

    print("\n>>>> 回测已完成，开始执行 reward 自动调参与图表报告生成 ... <<<<")
    subprocess.run(cmd, check=False)


def _parse_action(decision: str) -> tuple[str, float]:
    raw = str(decision or "HOLD").upper().strip()
    if raw.startswith("BUY"):
        action = "BUY"
    elif raw.startswith("SELL"):
        action = "SELL"
    else:
        action = "HOLD"

    position = 1.0
    cleaned = raw.replace("BUY", "").replace("SELL", "").replace("%", "").strip()
    if cleaned:
        try:
            position = max(0.0, min(1.0, float(cleaned) / 100.0))
        except ValueError:
            position = 1.0

    return action, position


def save_run_summary(ticker: str, backtest_limit: int | None, run_rows: list[dict], auto_tune_enabled: bool):
    if not run_rows:
        return

    summary_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "monitoring", "backtest_runs")
    os.makedirs(summary_dir, exist_ok=True)

    total_trades = len([row for row in run_rows if row["action"] in {"BUY", "SELL"}])
    avg_real_pnl = sum(row["real_pnl"] for row in run_rows) / len(run_rows)
    avg_effective_pnl = sum(row["effective_pnl"] for row in run_rows) / len(run_rows)
    win_rate = len([row for row in run_rows if row["effective_pnl"] > 0]) / len(run_rows)

    summary = {
        "ticker": ticker,
        "backtest_limit": backtest_limit,
        "auto_tune_enabled": auto_tune_enabled,
        "total_days": len(run_rows),
        "total_trades": total_trades,
        "avg_real_pnl": avg_real_pnl,
        "avg_effective_pnl": avg_effective_pnl,
        "win_rate": win_rate,
        "start_date": run_rows[0]["date"],
        "end_date": run_rows[-1]["date"],
        "rows": run_rows,
    }

    file_path = os.path.join(summary_dir, f"{ticker}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n📊 回测监控摘要已保存至: {file_path}")


def fetch_external_knowledge(ticker):
    """并发拉取新闻与研报，减少 IO 等待。"""
    ticker = str(ticker).strip().zfill(6)
    real_news_kb = []

    def fetch_news():
        news_items = []
        news_df = ak.stock_news_em(symbol=ticker)
        for _, row in news_df.head(100).iterrows():
            pub_time = str(row.get('发布时间', '2000-01-01'))
            try:
                date_int = int(pub_time[:10].replace('-', ''))
            except:
                date_int = 20000101
            news_items.append({
                "page_content": "[外围资讯] " + str(row['新闻标题']) + " : " + str(row['新闻内容']),
                "metadata": {"date_int": date_int, "ticker": ticker, "source": "news"}
            })
        return news_items, len(news_df)

    def fetch_reports():
        report_items = []
        report_df = ak.stock_research_report_em(symbol=ticker)
        if report_df is not None and not report_df.empty:
            for _, row in report_df.iterrows():
                pub_time = str(row.get('日期', '2000-01-01'))
                try:
                    date_int = int(pub_time[:10].replace('-', ''))
                except:
                    date_int = 20000101
                content = f"[券商研报] 机构: {row.get('机构', '未知')} | 评级: {row.get('东财评级', '未知')} | 核心观点摘要: {row.get('报告名称', '')}"
                report_items.append(
                    {
                        "page_content": content,
                        "metadata": {"date_int": date_int, "ticker": ticker, "source": "report"},
                    }
                )
        return report_items, 0 if report_df is None else len(report_df)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_news = executor.submit(fetch_news)
        future_reports = executor.submit(fetch_reports)

        try:
            news_items, news_count = future_news.result()
            real_news_kb.extend(news_items)
            print(f"成功攫取 {news_count} 条关于 {ticker} 的近期新闻。")
        except Exception as e:
            print(f"新闻抓取受限: {e}")

        try:
            report_items, report_count = future_reports.result()
            real_news_kb.extend(report_items)
            print(f"成功攫取 {report_count} 份专业研报注入向知识库。")
        except Exception as e:
            print(f"研报提取遭遇阻碍: {e}")

    return real_news_kb

def main():
    args = parse_args()
    print("=== 启动 A股智能投研多智能体系统：全年历史回测模式 ==================")
    print("本回测将跨越过去约3年（全真实历史数据）。")
    print("以天为单位滑动，收集每日特征 -> 做出交易决策 -> T+1日结算盈亏存入经验库。")
    print("========================================================================\n")

    # 1. 载入全局长期记忆突触
    memory_bank = MemoryBank()

    if args.ticker:
        ticker = str(args.ticker).strip().zfill(6)
        backtest_limit = args.backtest_limit
    else:
        try:
            ticker = ""
            while not ticker:
                ticker = input("请输入您要分析的 A股 股票代码 (例如: 600519，000001 等): ").strip()
                if not ticker:
                    print("股票代码不能为空，请重新输入。")
            
            if args.backtest_limit is not None:
                backtest_limit = args.backtest_limit
            else:
                backtest_days_input = input("请输入您想回测的最近交易日天数 (直接回车默认一整年): ").strip()
                if backtest_days_input and backtest_days_input.isdigit():
                    backtest_limit = int(backtest_days_input)
                else:
                    backtest_limit = None
        except (EOFError, KeyboardInterrupt):
            print("\n程序已取消。")
            return

    ticker = str(ticker).strip().zfill(6)
    print(f"\n>>>> 正在从 AKShare 拉取 [{ticker}] 的过去一整年K线数据进行切片循环 ... <<<<")
    
    # 获取接近 1 年的日线行情数据（以 250 个交易日约1年计，我们将拉去稍多一些方便计算初始特征）
    try:
        # 获取最近 300 个交易日以涵盖整个前一年并留有10天特征冗余
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
    print(f"[研报与深度评级] 正在获取 {ticker} 历史券商研报补充认知...")
    real_news_kb = fetch_external_knowledge(ticker)

    if not real_news_kb:
        print("未抓取到任何外部文本，使用内置降级备用认知。")
        real_news_kb = [
            {
                "page_content": f"{ticker} 暂未抓到外部新闻，维持中性观察并降低新闻因子权重。",
                "metadata": {"date_int": 20991231, "ticker": ticker, "source": "fallback"},
            }
        ]
        
    print(f"✅ RAG 混合语料筹备完毕，总计向 ChromaDB 灌入 {len(real_news_kb)} 条高维投研文本。")

    rag_engine = SimpleRAG(data_sources=real_news_kb)
    
    # == 实例化全链路多智能体团队 ==
    # 1. 两个综合分析师：技术+主力资金、基本面+新闻研报
    technical_flow_analyst = TechnicalFlowAnalyst(name="技术资金综合分析师")
    fundamental_news_analyst = FundamentalNewsAnalyst(name="基本面新闻综合分析师", rag_engine=rag_engine)
    
    # 2. 裁判博弈层与执行层
    referee = GameReferee(name="无情裁判官", memory_bank=memory_bank)
    risk_manager = RiskManager(name="风控大脑", memory_bank=memory_bank)
    trader_agent = TraderAgent(name="极速交易接口")
    reflector_agent = QuantitativeRiskReflector(name="量化策略迭代官", memory_bank=memory_bank)
    
    start_index = 60 # 从已经过拟合/训练完毕后的第60天起开始向后滚动盘面
    total_days = len(df_hist) - 1 # 留最后一天给 T+1 用
    run_rows = []

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
        
        # 阶段1: 两个综合分析师并行输出多空比率
        print("  [System] 启动两个综合分析师并发执行 (Async Threads)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_technical_flow = executor.submit(
                technical_flow_analyst.step,
                ticker,
                features=features,
                target_date=str(target_date),
            )
            future_fundamental_news = executor.submit(
                fundamental_news_analyst.step,
                ticker,
                target_date=str(target_date),
            )

            r_technical_flow = future_technical_flow.result()
            r_fundamental_news = future_fundamental_news.result()
        print("  [System] 两个综合分析师完成多空比率判断。")
        
        all_reports = [r_technical_flow, r_fundamental_news]
        
        # 阶段2: 两个分析师与裁判官的有限深度博弈
        referee_decision = referee.step_agent_game(
            analyst_a=technical_flow_analyst,
            analyst_b=fundamental_news_analyst,
            case_a=r_technical_flow,
            case_b=r_fundamental_news,
            ticker=ticker,
            max_depth=args.debate_depth,
            human_comment=args.human_comment,
            human_decision=args.human_decision,
        )
        
        # 阶段3: 风控与执行
        final_instruction = risk_manager.step(ticker, referee_decision)
        decision = trader_agent.step(final_instruction)

        # 阶段4: 模拟 T+1 日，去取第二天（i+1 行）的真实收盘涨跌幅！
        next_day_date = df_hist.iloc[i+1]['日期']
        real_pnl = float(df_hist.iloc[i+1]['涨跌幅'])
        
        print(f"  [游标揭晓] 进入 T+1日 ({next_day_date})，市场真实收盘涨跌幅为: {real_pnl}%")
        
        # 迭代官完成记录与数学期望计算
        reflector_agent.step(ticker, decision, all_reports, pnl_percent=real_pnl)

        action, position = _parse_action(decision)
        effective_pnl = real_pnl * position if action == "BUY" else ((-real_pnl) * position if action == "SELL" else 0.0)
        run_rows.append({
            "date": target_date,
            "next_date": next_day_date,
            "decision": decision,
            "action": action,
            "position": position,
            "real_pnl": real_pnl,
            "effective_pnl": effective_pnl,
        })
        
        # 为了不刷屏太快，微微暂停 0.1秒
        time.sleep(0.1)

    save_run_summary(ticker, backtest_limit, run_rows, args.auto_tune)

    if args.auto_tune:
        run_auto_tune(args)
   
if __name__ == "__main__":
    main()
