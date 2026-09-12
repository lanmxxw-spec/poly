from __future__ import annotations

import logging

from .base import BaseAgent, Signal

logger = logging.getLogger("polymarket_bot.agents.arbitrage")


class ArbitrageAgent(BaseAgent):
    """For binary (Yes/No) markets, the two outcome prices should sum to ~1.0
    (minus a small spread). If they sum to meaningfully more or less, there's
    a structural mispricing: buy the underpriced side (or both sides if the
    sum is far enough below 1.0 to lock in a profit regardless of outcome).

    This is the closest thing to a "risk-free-ish" edge in this system — real
    slippage and fees erode it, so the threshold is intentionally conservative.
    """
    name = "arbitrage"

    def __init__(self, cfg: dict, gamma_client=None):
        super().__init__(cfg)
        self.min_mispricing_pct = float(cfg.get("min_mispricing_pct", 3.0))

    def analyze(self, markets: list) -> list[Signal]:
        if not self.enabled:
            return []
        signals: list[Signal] = []
        for market in markets:
            if len(market.outcomes) != 2 or len(market.outcome_prices) != 2:
                continue
            total = sum(market.outcome_prices)
            deviation_pct = (total - 1.0) * 100

            if abs(deviation_pct) < self.min_mispricing_pct:
                continue

            if deviation_pct < 0:
                # Both sides underpriced relative to 1.0 -> buy both, locking in edge.
                for outcome, token_id, price in zip(
                    market.outcomes, market.outcome_token_ids, market.outcome_prices
                ):
                    signals.append(
                        Signal(
                            market_id=market.market_id,
                            token_id=token_id,
                            outcome=outcome,
                            side="BUY",
                            confidence=min(0.6 + abs(deviation_pct) / 20.0, 0.9),
                            target_price=price,
                            reasoning=f"outcome prices sum to {total:.3f} ({deviation_pct:+.1f}% vs 1.0) — "
                                      f"underpriced pair, buying both sides",
                            source_agent=self.name,
                        )
                    )
            else:
                # Overpriced -> sell the more expensive side (only actionable if you hold it,
                # so in practice this mostly informs "don't buy here" rather than a new position).
                idx_expensive = max(range(2), key=lambda i: market.outcome_prices[i])
                signals.append(
                    Signal(
                        market_id=market.market_id,
                        token_id=market.outcome_token_ids[idx_expensive],
                        outcome=market.outcomes[idx_expensive],
                        side="SELL",
                        confidence=min(0.5 + abs(deviation_pct) / 20.0, 0.8),
                        target_price=market.outcome_prices[idx_expensive],
                        reasoning=f"outcome prices sum to {total:.3f} ({deviation_pct:+.1f}% vs 1.0) — "
                                  f"overpriced pair, avoid/fade the richer side",
                        source_agent=self.name,
                    )
                )
        return signals
