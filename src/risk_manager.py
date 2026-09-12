"""Hard risk limits. These are checked *after* the coordinator decides what it
wants to trade, and can veto or downsize any trade. Nothing bypasses this layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .state_store import BotState

logger = logging.getLogger("polymarket_bot.risk")


@dataclass
class TradeDecision:
    market_id: str
    token_id: str
    outcome: str
    side: str
    price: float
    confidence: str
    size_usdc: float
    reasoning: str


class RiskManager:
    def __init__(self, cfg: dict, starting_bankroll_usdc: float):
        self.cfg = cfg
        # Only used to seed state.current_bankroll_usdc on the very first run.
        self.starting_bankroll = starting_bankroll_usdc
        self.max_risk_per_trade_pct = float(cfg["max_risk_per_trade_pct"])
        self.max_total_open_risk_pct = float(cfg["max_total_open_risk_pct"])
        self.daily_loss_limit_pct = float(cfg["daily_loss_limit_pct"])
        self.max_open_positions = int(cfg["max_open_positions"])
        self.min_confidence_to_trade = float(cfg["min_confidence_to_trade"])
        self.default_min_order_usdc = float(cfg.get("default_min_order_usdc", 1.0))

    def current_bankroll(self, state: BotState) -> float:
        """The live, compounding bankroll: starting seed + all realized P&L to
        date. Falls back to the configured starting value on the first ever run.
        """
        if state.current_bankroll_usdc is None:
            state.current_bankroll_usdc = self.starting_bankroll
        return state.current_bankroll_usdc

    def can_trade_at_all(self, state: BotState, kill_switch: bool) -> tuple[bool, str]:
        if kill_switch:
            return False, "kill_switch is enabled in config"
        if state.daily_loss_limit_hit:
            return False, "daily loss limit already hit today"
        bankroll = self.current_bankroll(state)
        loss_limit_usdc = bankroll * self.daily_loss_limit_pct / 100
        if state.realized_pnl_today_usdc <= -loss_limit_usdc:
            state.daily_loss_limit_hit = True
            return False, f"daily loss limit breached ({state.realized_pnl_today_usdc:.2f} USDC)"
        if len(state.positions) >= self.max_open_positions:
            return False, f"max open positions reached ({self.max_open_positions})"
        if bankroll <= 0:
            return False, f"bankroll depleted ({bankroll:.2f} USDC) — kill switch yourself and review"
        return True, "ok"

    def size_and_approve(
        self, aggregated_confidence: float, state: BotState, min_order_usdc: float | None = None
    ) -> tuple[bool, float, str]:
        """Returns (approved, size_usdc, reason). Sizing is a % of the CURRENT
        (compounding) bankroll, not the original config seed — this is what
        makes position sizes grow as the account grows, and shrink if it drops.
        """
        if aggregated_confidence < self.min_confidence_to_trade:
            return False, 0.0, f"confidence {aggregated_confidence:.2f} below threshold {self.min_confidence_to_trade}"

        bankroll = self.current_bankroll(state)
        per_trade_cap = bankroll * self.max_risk_per_trade_pct / 100
        total_cap = bankroll * self.max_total_open_risk_pct / 100
        remaining_room = total_cap - state.total_open_risk_usdc()

        if remaining_room <= 0:
            return False, 0.0, "max total open risk already allocated"

        # Scale size within [50%, 100%] of per-trade cap based on how far above
        # threshold the confidence is, then clamp to remaining portfolio room.
        confidence_scalar = min(
            1.0, 0.5 + (aggregated_confidence - self.min_confidence_to_trade) * 2
        )
        size = min(per_trade_cap * confidence_scalar, remaining_room)

        floor = min_order_usdc if min_order_usdc is not None else self.default_min_order_usdc
        if size < floor:
            return False, 0.0, f"computed size ${size:.2f} below this market's minimum order (${floor:.2f})"

        return True, round(size, 2), "approved"
