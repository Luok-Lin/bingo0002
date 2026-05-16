from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from math import tanh
from statistics import pstdev
from typing import Iterable, Sequence


DEFAULT_REWARD_CONFIG = {
    "return_scale": 5.0,
    "return_multiplier": 0.9,
    "direction_bonus": 0.05,
    "direction_penalty": 0.05,
    "volatility_penalty_scale": 0.04,
    "volatility_penalty_weight": 0.18,
    "drawdown_penalty_scale": 0.06,
    "drawdown_penalty_weight": 0.42,
    "turnover_penalty_base": 0.018,
    "turnover_penalty_position_weight": 0.045,
    "exposure_penalty_weight": 0.03,
    "loss_streak_penalty_window": 4.0,
    "loss_streak_penalty_weight": 0.12,
    "hold_penalty_scale": 5.0,
    "hold_penalty_weight": 0.15,
    "hold_streak_penalty_threshold": 3.0,
    "hold_streak_penalty_weight": 0.05,
    "correct_direction_reward_multiplier": 1.5,
    "reward_floor": -1.5,
    "reward_ceiling": 1.5,
}

_REWARD_CONFIG_CACHE: dict | None = None
_REWARD_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "reward_config.json")


def load_reward_config(config_path: str | None = None) -> dict:
    path = config_path or _REWARD_CONFIG_PATH
    merged = DEFAULT_REWARD_CONFIG.copy()

    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                merged.update(loaded)
    except Exception:
        pass

    return merged


def get_reward_config() -> dict:
    global _REWARD_CONFIG_CACHE
    if _REWARD_CONFIG_CACHE is None:
        _REWARD_CONFIG_CACHE = load_reward_config()
    return _REWARD_CONFIG_CACHE.copy()


def _cfg(config: dict | None, key: str) -> float:
    source = config or get_reward_config()
    return float(source.get(key, DEFAULT_REWARD_CONFIG[key]))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_percent_series(values: Sequence[float] | None) -> list[float]:
    if not values:
        return []

    normalized: list[float] = []
    for raw in values:
        if raw is None:
            continue
        normalized.append(float(raw) / 100.0)
    return normalized


def _max_drawdown(returns: Iterable[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0

    for item in returns:
        equity *= 1.0 + float(item)
        if equity > peak:
            peak = equity
        if peak > 0:
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return max_drawdown


def _loss_streak(returns: Sequence[float]) -> int:
    streak = 0
    for value in reversed(returns):
        if value < 0:
            streak += 1
        else:
            break
    return streak


def _current_hold_streak(recent_actions: Sequence[str] | None, current_action: str) -> int:
    if current_action != "HOLD":
        return 0

    streak = 1
    for action in reversed(recent_actions or []):
        if str(action or "HOLD").upper().strip().startswith("HOLD"):
            streak += 1
        else:
            break
    return streak


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    return_component: float
    direction_bonus: float
    volatility_penalty: float
    drawdown_penalty: float
    turnover_penalty: float
    exposure_penalty: float
    loss_streak_penalty: float
    hold_penalty: float
    hold_streak_penalty: float
    recent_volatility: float
    recent_max_drawdown: float
    recent_loss_streak: int
    recent_hold_streak: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_trade_reward(
    action: str,
    actual_pnl_percent: float,
    market_move_percent: float,
    historical_pnls: Sequence[float] | None = None,
    recent_actions: Sequence[str] | None = None,
    position: float = 1.0,
    config: dict | None = None,
) -> RewardBreakdown:
    """将收益、风险与行为成本统一映射为一个可学习的 reward。"""

    action = (action or "HOLD").upper().strip()
    history = _normalize_percent_series(historical_pnls)

    return_component = tanh(float(actual_pnl_percent) / _cfg(config, "return_scale")) * _cfg(config, "return_multiplier")
    recent_volatility = pstdev(history) if len(history) > 1 else 0.0
    recent_max_drawdown = _max_drawdown(history + [float(actual_pnl_percent) / 100.0]) if history else abs(float(actual_pnl_percent)) / 100.0
    recent_loss_streak = _loss_streak(history)
    recent_hold_streak = _current_hold_streak(recent_actions, action)

    direction_bonus = 0.0
    if action in {"BUY", "SELL"}:
        if actual_pnl_percent > 0:
            direction_bonus = _cfg(config, "direction_bonus") * _cfg(config, "correct_direction_reward_multiplier")
        elif actual_pnl_percent < 0:
            direction_bonus = -_cfg(config, "direction_penalty")

    volatility_penalty = _clamp(recent_volatility / _cfg(config, "volatility_penalty_scale"), 0.0, 1.0) * _cfg(config, "volatility_penalty_weight")
    drawdown_penalty = _clamp(recent_max_drawdown / _cfg(config, "drawdown_penalty_scale"), 0.0, 1.0) * _cfg(config, "drawdown_penalty_weight")
    turnover_penalty = _cfg(config, "turnover_penalty_base") + _clamp(position, 0.0, 1.0) * _cfg(config, "turnover_penalty_position_weight") if action in {"BUY", "SELL"} else 0.0
    exposure_penalty = _clamp(position, 0.0, 1.0) * _cfg(config, "exposure_penalty_weight") if action in {"BUY", "SELL"} else 0.0
    loss_streak_penalty = _clamp(recent_loss_streak / _cfg(config, "loss_streak_penalty_window"), 0.0, 1.0) * _cfg(config, "loss_streak_penalty_weight") if action in {"BUY", "SELL"} else 0.0
    hold_penalty = _clamp(abs(float(market_move_percent)) / _cfg(config, "hold_penalty_scale"), 0.0, 1.0) * _cfg(config, "hold_penalty_weight") if action == "HOLD" else 0.0
    hold_streak_penalty = 0.0
    if action == "HOLD":
        hold_threshold = _cfg(config, "hold_streak_penalty_threshold")
        if recent_hold_streak >= hold_threshold:
            hold_streak_ratio = (recent_hold_streak - hold_threshold + 1.0) / max(hold_threshold, 1.0)
            hold_streak_penalty = _clamp(hold_streak_ratio, 0.0, 1.0) * _cfg(config, "hold_streak_penalty_weight")

    reward = (
        return_component
        + direction_bonus
        - volatility_penalty
        - drawdown_penalty
        - turnover_penalty
        - exposure_penalty
        - loss_streak_penalty
        - hold_penalty
        - hold_streak_penalty
    )

    reward = _clamp(reward, _cfg(config, "reward_floor"), _cfg(config, "reward_ceiling"))

    return RewardBreakdown(
        reward=reward,
        return_component=return_component,
        direction_bonus=direction_bonus,
        volatility_penalty=volatility_penalty,
        drawdown_penalty=drawdown_penalty,
        turnover_penalty=turnover_penalty,
        exposure_penalty=exposure_penalty,
        loss_streak_penalty=loss_streak_penalty,
        hold_penalty=hold_penalty,
        hold_streak_penalty=hold_streak_penalty,
        recent_volatility=recent_volatility,
        recent_max_drawdown=recent_max_drawdown,
        recent_loss_streak=recent_loss_streak,
        recent_hold_streak=recent_hold_streak,
    )