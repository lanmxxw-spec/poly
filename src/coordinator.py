"""Aggregates raw Signal objects from every agent into one decision per
(market, outcome). Multiple agents agreeing on the same side raises combined
confidence (confluence); agents disagreeing on the same outcome cancel out.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from .agents.base import Signal
from .risk_manager import RiskManager, TradeDecision
from .state_store import BotState

logger = logging.getLogger("polymarket_bot.coordinator")

# Each agent's opinion is weighted — sentiment (Claude-reasoned) and arbitrage
# (near-mechanical) are trusted more than the purely technical agents.
AGENT_WEIGHTS = {
    "sentiment": 1.6,
    "arbitrage": 1.3,
    "momentum": 0.6,
    "volume_spike": 0.4,
}


class PortfolioCoordinator:
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager

    def aggregate(self, all_signals: list[Signal]) -> list[TradeDecision]:
        grouped: dict[tuple[str, str], list[Signal]] = defaultdict(list)
        for sig in all_signals:
            grouped[(sig.market_id, sig.token_id)].append(sig)

        decisions: list[TradeDecision] = []
        for (market_id, token_id), sigs in grouped.items():
            buy_score = sum(
                s.confidence * AGENT_WEIGHTS.get(s.source_agent, 1.0)
                for s in sigs if s.side == "BUY"
            )
            sell_score = sum(
                s.confidence * AGENT_WEIGHTS.get(s.source_agent, 1.0)
                for s in sigs if s.side == "SELL"
            )
            total_weight = sum(AGENT_WEIGHTS.get(s.source_agent, 1.0) for s in sigs)
            if total_weight == 0:
                continue

            net_score = (buy_score - sell_score) / total_weight
            side = "BUY" if net_score > 0 else "SELL"
            aggregated_confidence = min(abs(net_score), 1.0)

            contributing = [s for s in sigs if s.side == side]
            if not contributing:
                continue
            avg_target_price = sum(s.target_price for s in contributing) / len(contributing)
            reasons = "; ".join(f"[{s.source_agent}] {s.reasoning}" for s in contributing)
            outcome_name = contributing[0].outcome

            decisions.append(
                TradeDecision(
                    market_id=market_id,
                    token_id=token_id,
                    outcome=outcome_name,
                    side=side,
                    price=avg_target_price,
                    confidence=f"{aggregated_confidence:.2f} (from {len(sigs)} signal(s), "
                               f"{len(contributing)} agreeing)",
                    size_usdc=0.0,  # filled in below after risk sizing
                    reasoning=reasons,
                )
            )
        return decisions

    def finalize_with_risk(
        self, decisions: list[TradeDecision], state: BotState, min_order_by_market: dict[str, float] | None = None
    ) -> list[TradeDecision]:
        approved: list[TradeDecision] = []
        min_order_by_market = min_order_by_market or {}
        # Sort by confidence descending so the best ideas get first claim on risk budget.
        decisions_sorted = sorted(
            decisions, key=lambda d: float(d.confidence.split(" ")[0]), reverse=True
        )
        for d in decisions_sorted:
            confidence_value = float(d.confidence.split(" ")[0])
            min_order = min_order_by_market.get(d.market_id)
            ok, size, reason = self.risk_manager.size_and_approve(confidence_value, state, min_order)
            if not ok:
                logger.info("Rejected %s %s on market %s: %s", d.side, d.outcome, d.market_id, reason)
                continue
            d.size_usdc = size
            approved.append(d)
        return approved
