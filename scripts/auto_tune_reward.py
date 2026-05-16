from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from itertools import product
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import optuna
except Exception:
    optuna = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rl.reward import compute_trade_reward, load_reward_config


DEFAULT_LOG_PATH = os.path.join(BASE_DIR, "data", "json", "reflections.json")
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "rl", "reward_config.json")


@dataclass
class EvaluationResult:
    score: float
    validation_score: float | None
    total_reward: float
    avg_reward: float
    reward_std: float
    win_rate: float
    avg_positive_reward: float
    avg_negative_reward: float
    max_drawdown: float
    trade_count: int
    buy_sell_count: int
    hold_count: int
    trade_rate: float
    avg_trade_reward: float
    avg_hold_reward: float
    cumulative_effective_pnl: float
    avg_effective_pnl: float
    directional_accuracy: float
    missed_opportunity_rate: float
    manual_feedback_count: int
    rewards: list[float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto tune reward_config.json from historical backtest logs.")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="Path to reflections.json")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH, help="Path to reward_config.json")
    parser.add_argument("--output-path", default=None, help="Where to write the tuned config. Defaults to config-path")
    parser.add_argument("--report-dir", default=None, help="Directory to store chart report files")
    parser.add_argument("--window", type=int, default=120, help="Use the most recent N records for tuning")
    parser.add_argument("--samples", type=int, default=80, help="How many random candidates to try")
    parser.add_argument("--grid", action="store_true", help="Use a small deterministic grid around baseline")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Use Optuna if available; specify number of trials")
    parser.add_argument("--optuna-study-name", default="reward_tuning", help="Optuna study name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--validation-ratio", type=float, default=0.3, help="Hold out the latest ratio of records for validation")
    parser.add_argument("--dry-run", action="store_true", help="Do not write tuned config to disk")
    return parser.parse_args()


def _load_records(log_path: str) -> list[dict]:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"找不到回测日志: {log_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("回测日志格式错误：顶层应为列表")

    return records


def _normalize_decision(record: dict) -> tuple[str, float]:
    raw = str(record.get("decision", "HOLD") or "HOLD").strip().upper()
    if raw.startswith("BUY"):
        action = "BUY"
    elif raw.startswith("SELL"):
        action = "SELL"
    else:
        action = "HOLD"

    position = float(record.get("position", record.get("position_percent", 1.0)) or 1.0)
    if position > 1.0:
        position = position / 100.0
    position = max(0.0, min(1.0, position))
    for token in [raw.replace("BUY", ""), raw.replace("SELL", "")]:
        token = token.replace("%", "").strip()
        if token:
            try:
                position = max(0.0, min(1.0, float(token) / 100.0))
                break
            except ValueError:
                continue

    return action, position


def _market_move_percent(record: dict) -> float:
    for key in ("market_move_percent", "real_pnl", "market_pnl_percent"):
        if key in record and record.get(key) is not None:
            return float(record.get(key) or 0.0)
    return float(record.get("pnl_percent", 0.0) or 0.0)


def _actual_pnl_percent(record: dict, action: str, position: float, market_move_percent: float) -> float:
    if "pnl_percent" in record and record.get("pnl_percent") is not None:
        return float(record.get("pnl_percent") or 0.0)
    if action == "BUY":
        return market_move_percent * position
    if action == "SELL":
        return -market_move_percent * position
    return 0.0


def _max_drawdown_from_rewards(rewards: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for reward in rewards:
        equity *= max(0.01, 1.0 + reward / 10.0)
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak else 0.0
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def _max_drawdown_from_pnls(pnls: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for pnl in pnls:
        equity *= max(0.01, 1.0 + float(pnl) / 100.0)
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak else 0.0
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def _evaluate_config(records: list[dict], config: dict) -> EvaluationResult:
    historical_pnls: list[float] = []
    recent_actions_by_ticker: dict[str, list[str]] = {}
    rewards: list[float] = []
    trade_rewards: list[float] = []
    hold_rewards: list[float] = []
    effective_pnls: list[float] = []
    positive_rewards = []
    negative_rewards = []
    buy_sell_count = 0
    hold_count = 0
    correct_direction_count = 0
    directional_sample_count = 0
    missed_opportunity_count = 0
    hold_opportunity_count = 0
    manual_feedback_count = 0

    for record in records:
        action, position = _normalize_decision(record)
        market_move_percent = _market_move_percent(record)
        actual_pnl_percent = _actual_pnl_percent(record, action, position, market_move_percent)
        ticker = str(record.get("ticker", "__GLOBAL__") or "__GLOBAL__")
        recent_actions = recent_actions_by_ticker.setdefault(ticker, [])
        reflection_text = str(record.get("reflection_text", "") or "")

        if "人工" in reflection_text or str(record.get("role", "")).lower() == "human":
            manual_feedback_count += 1

        if action in {"BUY", "SELL"}:
            buy_sell_count += 1
            if abs(market_move_percent) > 0:
                directional_sample_count += 1
                if (action == "BUY" and market_move_percent > 0) or (action == "SELL" and market_move_percent < 0):
                    correct_direction_count += 1
        else:
            hold_count += 1
            if abs(market_move_percent) >= 0.5:
                hold_opportunity_count += 1
                missed_opportunity_count += 1

        breakdown = compute_trade_reward(
            action=action,
            actual_pnl_percent=actual_pnl_percent,
            market_move_percent=market_move_percent,
            historical_pnls=historical_pnls,
            recent_actions=recent_actions,
            position=position,
            config=config,
        )

        reward = float(breakdown.reward)
        rewards.append(reward)
        historical_pnls.append(actual_pnl_percent)
        recent_actions.append(action)
        effective_pnls.append(actual_pnl_percent)
        if action in {"BUY", "SELL"}:
            trade_rewards.append(reward)
        else:
            hold_rewards.append(reward)

        if reward > 0:
            positive_rewards.append(reward)
        elif reward < 0:
            negative_rewards.append(reward)

    trade_count = len(rewards)
    avg_reward = mean(rewards) if rewards else 0.0
    reward_std = pstdev(rewards) if len(rewards) > 1 else 0.0
    win_rate = len([r for r in rewards if r > 0]) / trade_count if trade_count else 0.0
    avg_positive_reward = mean(positive_rewards) if positive_rewards else 0.0
    avg_negative_reward = mean(negative_rewards) if negative_rewards else 0.0
    max_drawdown = _max_drawdown_from_rewards(rewards)
    pnl_drawdown = _max_drawdown_from_pnls(effective_pnls)
    trade_rate = buy_sell_count / trade_count if trade_count else 0.0
    avg_trade_reward = mean(trade_rewards) if trade_rewards else 0.0
    avg_hold_reward = mean(hold_rewards) if hold_rewards else 0.0
    cumulative_effective_pnl = sum(effective_pnls)
    avg_effective_pnl = mean(effective_pnls) if effective_pnls else 0.0
    directional_accuracy = correct_direction_count / directional_sample_count if directional_sample_count else 0.0
    missed_opportunity_rate = missed_opportunity_count / hold_opportunity_count if hold_opportunity_count else 0.0

    # 目标函数：同时优化 reward、真实有效收益、方向正确率，并压制长期 HOLD 与错过机会。
    score = (
        avg_reward * 1.4
        + win_rate * 0.8
        + avg_effective_pnl * 0.18
        + cumulative_effective_pnl * 0.015
        + directional_accuracy * 1.1
        + trade_rate * 0.45
        + avg_positive_reward * 0.5
        + avg_trade_reward * 0.35
        + avg_negative_reward * 0.75  # negative 为负数，所以会惩罚
        - reward_std * 1.0
        - max_drawdown * 1.8
        - pnl_drawdown * 2.0
        - missed_opportunity_rate * 1.2
    )

    score += math.log1p(max(0, buy_sell_count)) * 0.08
    score -= hold_count * 0.004

    return EvaluationResult(
        score=score,
        validation_score=None,
        total_reward=sum(rewards),
        avg_reward=avg_reward,
        reward_std=reward_std,
        win_rate=win_rate,
        avg_positive_reward=avg_positive_reward,
        avg_negative_reward=avg_negative_reward,
        max_drawdown=max_drawdown,
        trade_count=trade_count,
        buy_sell_count=buy_sell_count,
        hold_count=hold_count,
        trade_rate=trade_rate,
        avg_trade_reward=avg_trade_reward,
        avg_hold_reward=avg_hold_reward,
        cumulative_effective_pnl=cumulative_effective_pnl,
        avg_effective_pnl=avg_effective_pnl,
        directional_accuracy=directional_accuracy,
        missed_opportunity_rate=missed_opportunity_rate,
        manual_feedback_count=manual_feedback_count,
        rewards=rewards,
    )


def _grid_candidates(base: dict) -> list[dict]:
    axes = {
        "return_scale": [0.9, 1.0, 1.1],
        "return_multiplier": [0.85, 0.9, 0.95],
        "volatility_penalty_weight": [0.85, 1.0, 1.15],
        "drawdown_penalty_weight": [0.85, 1.0, 1.15],
        "turnover_penalty_base": [0.85, 1.0, 1.15],
        "exposure_penalty_weight": [0.85, 1.0, 1.15],
        "loss_streak_penalty_weight": [0.75, 1.0, 1.25],
        "hold_penalty_weight": [0.9, 1.0, 1.2],
        "hold_streak_penalty_weight": [0.8, 1.0, 1.3],
        "correct_direction_reward_multiplier": [1.0, 1.2, 1.5],
    }

    keys = list(axes.keys())
    candidates = []
    for multipliers in product(*(axes[key] for key in keys)):
        candidate = copy.deepcopy(base)
        for key, factor in zip(keys, multipliers):
            candidate[key] = round(float(candidate[key]) * float(factor), 6)
        candidates.append(candidate)
    return candidates


def _suggest_candidate_from_trial(trial, base: dict) -> dict:
    candidate = copy.deepcopy(base)
    candidate["return_scale"] = trial.suggest_float("return_scale", 3.5, 7.0)
    candidate["return_multiplier"] = trial.suggest_float("return_multiplier", 0.75, 1.05)
    candidate["direction_bonus"] = trial.suggest_float("direction_bonus", 0.03, 0.08)
    candidate["direction_penalty"] = trial.suggest_float("direction_penalty", 0.03, 0.08)
    candidate["volatility_penalty_weight"] = trial.suggest_float("volatility_penalty_weight", 0.10, 0.28)
    candidate["drawdown_penalty_weight"] = trial.suggest_float("drawdown_penalty_weight", 0.22, 0.62)
    candidate["turnover_penalty_base"] = trial.suggest_float("turnover_penalty_base", 0.01, 0.04)
    candidate["turnover_penalty_position_weight"] = trial.suggest_float("turnover_penalty_position_weight", 0.02, 0.07)
    candidate["exposure_penalty_weight"] = trial.suggest_float("exposure_penalty_weight", 0.01, 0.06)
    candidate["loss_streak_penalty_weight"] = trial.suggest_float("loss_streak_penalty_weight", 0.06, 0.22)
    candidate["hold_penalty_weight"] = trial.suggest_float("hold_penalty_weight", 0.08, 0.22)
    candidate["hold_streak_penalty_weight"] = trial.suggest_float("hold_streak_penalty_weight", 0.02, 0.10)
    candidate["correct_direction_reward_multiplier"] = trial.suggest_float("correct_direction_reward_multiplier", 1.0, 1.8)
    return candidate


def _optuna_candidates(base: dict, records: list[dict], trials: int, seed: int, study_name: str) -> tuple[list[dict], list[EvaluationResult], dict | None]:
    if optuna is None:
        return [], [], None

    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=optuna.samplers.TPESampler(seed=seed))
    candidate_configs: list[dict] = []
    candidate_results: list[EvaluationResult] = []

    def objective(trial):
        candidate = _suggest_candidate_from_trial(trial, base)
        result = _evaluate_config(records, candidate)
        candidate_configs.append(candidate)
        candidate_results.append(result)
        trial.set_user_attr("score", result.score)
        trial.set_user_attr("avg_reward", result.avg_reward)
        trial.set_user_attr("win_rate", result.win_rate)
        return result.score

    study.optimize(objective, n_trials=max(1, trials), show_progress_bar=False)
    best_params = study.best_trial.params if study.best_trial is not None else None
    return candidate_configs, candidate_results, best_params


def _random_candidates(base: dict, samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates = []
    for _ in range(samples):
        candidate = copy.deepcopy(base)
        for key in [
            "return_scale",
            "return_multiplier",
            "volatility_penalty_weight",
            "drawdown_penalty_weight",
            "turnover_penalty_base",
            "turnover_penalty_position_weight",
            "exposure_penalty_weight",
            "loss_streak_penalty_weight",
            "hold_penalty_weight",
            "hold_streak_penalty_weight",
            "correct_direction_reward_multiplier",
        ]:
            jitter = rng.uniform(0.82, 1.18)
            candidate[key] = round(float(candidate[key]) * jitter, 6)
        candidates.append(candidate)
    return candidates


def _format_metric_line(name: str, value: float) -> str:
    if value is None:
        return f"{name}=N/A"
    if name in {
        "score",
        "validation_score",
        "total_reward",
        "avg_reward",
        "reward_std",
        "avg_positive_reward",
        "avg_negative_reward",
        "max_drawdown",
        "avg_trade_reward",
        "avg_hold_reward",
        "cumulative_effective_pnl",
        "avg_effective_pnl",
    }:
        return f"{name}={value:.6f}"
    if name in {"win_rate", "trade_rate", "directional_accuracy", "missed_opportunity_rate"}:
        return f"{name}={value:.2%}"
    return f"{name}={value}"


def _split_train_validation(records: list[dict], validation_ratio: float) -> tuple[list[dict], list[dict]]:
    if len(records) < 10 or validation_ratio <= 0:
        return records, []

    ratio = max(0.05, min(0.5, validation_ratio))
    validation_size = max(1, int(len(records) * ratio))
    if validation_size >= len(records):
        validation_size = max(1, len(records) // 3)
    split_idx = len(records) - validation_size
    return records[:split_idx], records[split_idx:]


def _with_validation_score(result: EvaluationResult, validation_score: float | None) -> EvaluationResult:
    result.validation_score = validation_score
    return result


def _write_config(path: str, config: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _build_report_dir(report_dir: str | None) -> str:
    from datetime import datetime

    if report_dir:
        return _ensure_dir(report_dir)

    base_report_dir = os.path.join(BASE_DIR, "data", "reward_tuning_reports")
    timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _ensure_dir(os.path.join(base_report_dir, timestamp_dir))


def _plot_metric_comparison(report_dir: str, baseline: EvaluationResult, best: EvaluationResult) -> str:
    metrics = ["score", "avg_reward", "trade_rate", "directional_accuracy", "avg_effective_pnl", "max_drawdown"]
    baseline_values = [
        baseline.score,
        baseline.avg_reward,
        baseline.trade_rate,
        baseline.directional_accuracy,
        baseline.avg_effective_pnl,
        baseline.max_drawdown,
    ]
    best_values = [
        best.score,
        best.avg_reward,
        best.trade_rate,
        best.directional_accuracy,
        best.avg_effective_pnl,
        best.max_drawdown,
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(metrics)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], baseline_values, width=width, label="Baseline", color="#8A8A8A")
    ax.bar([i + width / 2 for i in x], best_values, width=width, label="Best", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Baseline vs Best Reward Metrics")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(report_dir, "metric_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_reward_distribution(report_dir: str, baseline: EvaluationResult, best: EvaluationResult) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(baseline.rewards, bins=20, alpha=0.55, label="Baseline", color="#8A8A8A", density=True)
    ax.hist(best.rewards, bins=20, alpha=0.55, label="Best", color="#1f77b4", density=True)
    ax.axvline(baseline.avg_reward, color="#8A8A8A", linestyle="--", linewidth=1)
    ax.axvline(best.avg_reward, color="#1f77b4", linestyle="--", linewidth=1)
    ax.set_title("Reward Distribution Comparison")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(report_dir, "reward_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_candidate_scores(report_dir: str, candidate_scores: list[float], best_score: float, baseline_score: float) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(candidate_scores, color="#1f77b4", linewidth=1.8, marker="o", markersize=3)
    ax.axhline(baseline_score, color="#8A8A8A", linestyle="--", label="Baseline")
    ax.axhline(best_score, color="#2ca02c", linestyle="--", label="Best")
    ax.set_title("Candidate Search Scores")
    ax.set_xlabel("Candidate Index")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(report_dir, "candidate_scores.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_config_changes(report_dir: str, baseline: dict, best: dict) -> str:
    keys = [
        "return_scale",
        "return_multiplier",
        "volatility_penalty_weight",
        "drawdown_penalty_weight",
        "turnover_penalty_base",
        "turnover_penalty_position_weight",
        "exposure_penalty_weight",
        "loss_streak_penalty_weight",
        "hold_penalty_weight",
        "hold_streak_penalty_weight",
        "correct_direction_reward_multiplier",
    ]
    deltas = []
    labels = []
    for key in keys:
        base_value = float(baseline.get(key, 0.0))
        best_value = float(best.get(key, 0.0))
        labels.append(key)
        if base_value == 0:
            deltas.append(0.0)
        else:
            deltas.append((best_value / base_value - 1.0) * 100.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in deltas]
    ax.barh(labels, deltas, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Reward Config Relative Changes (%)")
    ax.set_xlabel("Change vs Baseline (%)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(report_dir, "config_changes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    records = _load_records(args.log_path)
    if args.window and args.window > 0:
        records = records[-args.window :]
    train_records, validation_records = _split_train_validation(records, args.validation_ratio)

    baseline = load_reward_config(args.config_path)
    baseline_result = _evaluate_config(train_records, baseline)
    baseline_validation_result = _evaluate_config(validation_records, baseline) if validation_records else None
    _with_validation_score(
        baseline_result,
        baseline_validation_result.score if baseline_validation_result else None,
    )

    informative_trade_count = sum(1 for record in records if str(record.get("decision", "HOLD") or "HOLD").upper().startswith(("BUY", "SELL")))
    if informative_trade_count == 0:
        print("⚠️ 当前回测窗口里几乎没有 BUY/SELL 记录，reward 搜索很难产生区分度；建议换成有真实交易的回测日志再调参。")
    market_move_count = sum(1 for record in records if any(key in record for key in ("market_move_percent", "real_pnl", "market_pnl_percent")))
    if market_move_count == 0:
        print("⚠️ 当前日志缺少 market_move_percent/real_pnl，HOLD 错过机会率只能在未来新日志中准确评估。")

    if args.grid:
        candidates = _grid_candidates(baseline)
        optimizer_name = "grid"
        optuna_best_params = None
    elif args.optuna_trials and args.optuna_trials > 0 and optuna is not None:
        candidates, _, optuna_best_params = _optuna_candidates(baseline, train_records, args.optuna_trials, args.seed, args.optuna_study_name)
        optimizer_name = "optuna"
    elif args.optuna_trials and args.optuna_trials > 0 and optuna is None:
        print("⚠️ 检测到 --optuna-trials 但当前环境未安装 optuna，将自动回退到随机搜索。")
        candidates = [baseline] + _random_candidates(baseline, args.samples, args.seed)
        optimizer_name = "random"
        optuna_best_params = None
    else:
        candidates = [baseline] + _random_candidates(baseline, args.samples, args.seed)
        optimizer_name = "random"
        optuna_best_params = None

    best_config = baseline
    best_result = baseline_result
    best_validation_result = baseline_validation_result
    candidate_results: list[EvaluationResult] = []
    baseline_selection_score = (
        baseline_validation_result.score * 0.7 + baseline_result.score * 0.3
        if baseline_validation_result
        else baseline_result.score
    )
    best_selection_score = baseline_selection_score

    for idx, candidate in enumerate(candidates, start=1):
        result = _evaluate_config(train_records, candidate)
        validation_result = _evaluate_config(validation_records, candidate) if validation_records else None
        _with_validation_score(result, validation_result.score if validation_result else None)
        candidate_results.append(result)
        selection_score = (
            validation_result.score * 0.7 + result.score * 0.3
            if validation_result
            else result.score
        )
        if selection_score > best_selection_score:
            best_config = candidate
            best_result = result
            best_validation_result = validation_result
            best_selection_score = selection_score

    output_path = args.output_path or args.config_path
    report_dir = _build_report_dir(args.report_dir)

    print("=" * 72)
    print("Reward Auto Tuning Report")
    print("=" * 72)
    print(f"Records: total={len(records)}, train={len(train_records)}, validation={len(validation_records)}")
    print(f"Selection score: baseline={baseline_selection_score:.6f}, best={best_selection_score:.6f}")
    print()
    print("Baseline:")
    metric_fields = [
        "score",
        "validation_score",
        "total_reward",
        "avg_reward",
        "reward_std",
        "win_rate",
        "trade_rate",
        "avg_trade_reward",
        "avg_hold_reward",
        "cumulative_effective_pnl",
        "avg_effective_pnl",
        "directional_accuracy",
        "missed_opportunity_rate",
        "max_drawdown",
        "trade_count",
        "buy_sell_count",
        "hold_count",
    ]
    for field in metric_fields:
        print("  -", _format_metric_line(field, getattr(baseline_result, field)))
    if baseline_validation_result:
        print("  Validation:")
        for field in ["score", "avg_reward", "trade_rate", "cumulative_effective_pnl", "directional_accuracy", "missed_opportunity_rate", "max_drawdown"]:
            print("  -", _format_metric_line(field, getattr(baseline_validation_result, field)))
    print()
    print("Best:")
    for field in metric_fields:
        print("  -", _format_metric_line(field, getattr(best_result, field)))
    if best_validation_result:
        print("  Validation:")
        for field in ["score", "avg_reward", "trade_rate", "cumulative_effective_pnl", "directional_accuracy", "missed_opportunity_rate", "max_drawdown"]:
            print("  -", _format_metric_line(field, getattr(best_validation_result, field)))
    print()
    print("Delta:")
    print(f"  - selection_score_delta={best_selection_score - baseline_selection_score:.6f}")
    print(f"  - train_score_delta={best_result.score - baseline_result.score:.6f}")
    if best_validation_result and baseline_validation_result:
        print(f"  - validation_score_delta={best_validation_result.score - baseline_validation_result.score:.6f}")
    print(f"  - avg_reward_delta={best_result.avg_reward - baseline_result.avg_reward:.6f}")
    print(f"  - trade_rate_delta={best_result.trade_rate - baseline_result.trade_rate:.6f}")
    print(f"  - cumulative_effective_pnl_delta={best_result.cumulative_effective_pnl - baseline_result.cumulative_effective_pnl:.6f}")
    print(f"  - reward_std_delta={best_result.reward_std - baseline_result.reward_std:.6f}")
    print(f"  - max_drawdown_delta={best_result.max_drawdown - baseline_result.max_drawdown:.6f}")
    print()
    print("Tuned config:")
    print(json.dumps(best_config, ensure_ascii=False, indent=2))

    metric_chart = _plot_metric_comparison(report_dir, baseline_result, best_result)
    distribution_chart = _plot_reward_distribution(report_dir, baseline_result, best_result)
    candidate_scores = [
        (r.validation_score * 0.7 + r.score * 0.3) if r.validation_score is not None else r.score
        for r in candidate_results
    ]
    candidate_chart = _plot_candidate_scores(report_dir, candidate_scores, best_selection_score, baseline_selection_score)
    config_chart = _plot_config_changes(report_dir, baseline, best_config)

    report = {
        "log_path": args.log_path,
        "config_path": args.config_path,
        "output_path": output_path,
        "window": args.window,
        "validation_ratio": args.validation_ratio,
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "samples": args.samples,
        "grid": args.grid,
        "optimizer": optimizer_name,
        "optuna_trials": args.optuna_trials,
        "optuna_available": optuna is not None,
        "optuna_best_params": optuna_best_params,
        "dry_run": args.dry_run,
        "report_dir": report_dir,
        "baseline_selection_score": baseline_selection_score,
        "best_selection_score": best_selection_score,
        "baseline": baseline_result.__dict__,
        "baseline_validation": baseline_validation_result.__dict__ if baseline_validation_result else None,
        "best": best_result.__dict__,
        "best_validation": best_validation_result.__dict__ if best_validation_result else None,
        "baseline_config": baseline,
        "best_config": best_config,
        "charts": {
            "metric_comparison": metric_chart,
            "reward_distribution": distribution_chart,
            "candidate_scores": candidate_chart,
            "config_changes": config_chart,
        },
    }

    report_path = os.path.join(report_dir, "tuning_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(f"图表报告目录: {report_dir}")
    print(f"报告 JSON: {report_path}")
    print(f"图表文件: {metric_chart}")
    print(f"图表文件: {distribution_chart}")
    print(f"图表文件: {candidate_chart}")
    print(f"图表文件: {config_chart}")

    if not args.dry_run:
        _write_config(output_path, best_config)
        print()
        print(f"已写入: {output_path}")


if __name__ == "__main__":
    main()