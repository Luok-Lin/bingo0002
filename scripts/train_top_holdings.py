from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "个人_持股排名.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/backtest the top holdings from a ranking CSV.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="持股排名 CSV 路径")
    parser.add_argument("--top-n", type=int, default=10, help="选取前 N 只股票")
    parser.add_argument("--days", type=int, default=22, help="近一个月按 22 个交易日回测")
    parser.add_argument("--debate-depth", type=int, default=2, help="裁判与两个综合分析师最大博弈轮数")
    parser.add_argument("--tune-window", type=int, default=240, help="批量回测后 reward 自动调参窗口")
    parser.add_argument("--tune-samples", type=int, default=80, help="reward 自动调参随机候选数量")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令，不实际回测")
    parser.add_argument("--skip-auto-tune", action="store_true", help="跳过批量训练后的 reward 自动调参")
    return parser.parse_args()


def load_top_tickers(csv_path: str, top_n: int) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    holdings = []
    for row in rows[:top_n]:
        ticker = str(row.get("股票代码", "")).strip()
        if not ticker:
            continue
        holdings.append({
            "rank": row.get("排名", ""),
            "ticker": ticker.zfill(6),
            "name": row.get("股票名称", ""),
            "holder_count": row.get("持股股东数量", ""),
            "top_holder": row.get("最高涨幅股东", ""),
            "avg_30d_gain": row.get("30日平均涨幅", ""),
        })
    return holdings


def run_backtest(holding: dict, args: argparse.Namespace) -> dict:
    ticker = holding["ticker"]
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "main.py"),
        ticker,
        str(args.days),
        "--debate-depth",
        str(args.debate_depth),
    ]
    print(f"\n>>>>>>>>>>>>> 训练/回测 {ticker} {holding.get('name', '')} ({args.days}交易日) <<<<<<<<<<<<<")
    print("命令:", " ".join(cmd))

    if args.dry_run:
        return {**holding, "status": "dry_run", "returncode": None}

    result = subprocess.run(cmd, cwd=BASE_DIR)
    return {**holding, "status": "ok" if result.returncode == 0 else "failed", "returncode": result.returncode}


def run_auto_tune(args: argparse.Namespace) -> dict:
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "scripts", "auto_tune_reward.py"),
        "--window",
        str(args.tune_window),
        "--samples",
        str(args.tune_samples),
    ]
    print("\n>>>>>>>>>>>>> 批量回测完成，启动 reward 自动调参 <<<<<<<<<<<<<")
    print("命令:", " ".join(cmd))
    if args.dry_run:
        return {"status": "dry_run", "returncode": None}

    result = subprocess.run(cmd, cwd=BASE_DIR)
    return {"status": "ok" if result.returncode == 0 else "failed", "returncode": result.returncode}


def main() -> None:
    args = _parse_args()
    holdings = load_top_tickers(args.csv_path, args.top_n)
    if not holdings:
        raise RuntimeError(f"未能从 CSV 读取到股票: {args.csv_path}")

    print("=" * 72)
    print(f"Top {len(holdings)} 持股训练/回测任务")
    print("=" * 72)
    for item in holdings:
        print(f"{item['rank']}. {item['ticker']} {item['name']} | 股东数={item['holder_count']} | 30日均涨幅={item['avg_30d_gain']}")

    results = [run_backtest(item, args) for item in holdings]
    tune_result = None if args.skip_auto_tune else run_auto_tune(args)

    summary_dir = os.path.join(BASE_DIR, "data", "monitoring", "portfolio_training")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"top_holdings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    summary = {
        "csv_path": args.csv_path,
        "top_n": args.top_n,
        "days": args.days,
        "debate_depth": args.debate_depth,
        "results": results,
        "auto_tune": tune_result,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n训练/回测摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
