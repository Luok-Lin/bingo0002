from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    requested_action: str
    executed_action: str
    requested_position: float
    executed_position: float
    market_move_percent: float
    gross_pnl_percent: float
    cost_percent: float
    net_pnl_percent: float
    blocked_reason: str
    limit_percent: float

    def to_dict(self) -> dict:
        return asdict(self)


def infer_a_share_limit_percent(ticker: str, name: str = "") -> float:
    code = str(ticker or "").strip().zfill(6)
    display_name = str(name or "").upper()
    if "ST" in display_name:
        return 5.0
    if code.startswith(("300", "301", "688")):
        return 20.0
    return 10.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _limit_block_reason(action: str, current_change_percent: float, limit_percent: float) -> str:
    buffer = 0.05
    if action == "BUY" and current_change_percent >= limit_percent - buffer:
        return "涨停附近，按 A 股约束视为无法新开买入。"
    if action == "SELL" and current_change_percent <= -limit_percent + buffer:
        return "跌停附近，按 A 股约束视为无法卖出/做空。"
    return ""


def simulate_a_share_execution(
    ticker: str,
    action: str,
    position: float,
    market_move_percent: float,
    current_change_percent: float = 0.0,
    *,
    allow_short: bool = False,
    commission_rate: float = 0.0003,
    stamp_duty_rate: float = 0.0005,
    slippage_bps: float = 5.0,
    name: str = "",
) -> ExecutionResult:
    requested_action = str(action or "HOLD").upper().strip()
    if requested_action not in {"BUY", "SELL"}:
        requested_action = "HOLD"
    requested_position = max(0.0, min(1.0, _safe_float(position)))
    market_move = _safe_float(market_move_percent)
    current_change = _safe_float(current_change_percent)
    limit_percent = infer_a_share_limit_percent(ticker, name=name)

    blocked_reason = ""
    executed_action = requested_action
    executed_position = requested_position

    if requested_action == "SELL" and not allow_short:
        blocked_reason = "A 股长仓模式不执行做空，SELL 视为减仓/回避。"
        executed_action = "HOLD"
        executed_position = 0.0
    elif requested_action in {"BUY", "SELL"}:
        blocked_reason = _limit_block_reason(requested_action, current_change, limit_percent)
        if blocked_reason:
            executed_action = "HOLD"
            executed_position = 0.0

    if executed_action == "BUY":
        gross_pnl = market_move * executed_position
    elif executed_action == "SELL":
        gross_pnl = -market_move * executed_position
    else:
        gross_pnl = 0.0

    if executed_action in {"BUY", "SELL"}:
        slippage_rate = max(0.0, slippage_bps) / 10000.0
        # A-share stamp duty is charged on the sell leg. A round-trip execution
        # has one sell leg whether the signal starts with BUY or short SELL.
        tax_rate = max(0.0, stamp_duty_rate)
        round_trip_cost_rate = (2 * max(0.0, commission_rate)) + tax_rate + (2 * slippage_rate)
        cost_percent = round_trip_cost_rate * executed_position * 100.0
    else:
        cost_percent = 0.0

    net_pnl = gross_pnl - cost_percent
    return ExecutionResult(
        requested_action=requested_action,
        executed_action=executed_action,
        requested_position=round(requested_position, 4),
        executed_position=round(executed_position, 4),
        market_move_percent=round(market_move, 4),
        gross_pnl_percent=round(gross_pnl, 4),
        cost_percent=round(cost_percent, 4),
        net_pnl_percent=round(net_pnl, 4),
        blocked_reason=blocked_reason,
        limit_percent=limit_percent,
    )
