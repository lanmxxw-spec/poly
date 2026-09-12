from __future__ import annotations

import logging

from .base import BaseAgent, Signal

logger = logging.getLogger("polymarket_bot.agents.volume_spike")


class VolumeSpikeAgent(BaseAgent):
    """Flags markets where 24h volume is a large multiple of the market's
    typical liquidity, on the theory that a volume spike signals new
    information hitting the market before price has fully adjusted.

    This agent doesn't pick a direction on its own — it looks at which side
    of the current price the volume is concentrated by comparing the most
    recent price tick direction, and defers most of its confidence to
    corroboration from other agents (it mainly acts as a market filter/booster).

    Skips outcomes already priced near certainty (see MomentumAgent) — same
    bad risk/reward applies regardless of what's driving the volume.
    """
    name = "volume_spike"
    EXTREME_LOW = 0.03
    EXTREME_HIGH = 0.97

    def __init__(self, cfg: dict, gamma_client):
        super().__init__(cfg)
        self.gamma_client = gamma_client
        self.spike_multiple = float(cfg.get("spike_multiple", 3.0))

    def analyze(self, markets: list) -> list[Signal]:
        if not self.enabled:
            return []
        signals: list[Signal] = []
        for market in markets:
            if market.liquidity_usdc <= 0:
                continue
            ratio = market.volume_24h_usdc / market.liquidity_usdc
            if ratio < self.spike_multiple:
                continue

            for outcome, token_id, current_price in zip(
                market.outcomes, market.outcome_token_ids, market.outcome_prices
            ):
                if current_price <= self.EXTREME_LOW or current_price >= self.EXTREME_HIGH:
                    continue
                try:
                    history = self.gamma_client.fetch_price_history(token_id, interval="1h")
                except Exception as e:
                    logger.debug("Price history fetch failed for %s: %s", token_id, e)
                    continue
                if len(history) < 2:
                    continue
                recent_direction = float(history[-1]["p"]) - float(history[-2]["p"])
                if recent_direction == 0:
                    continue
                side = "BUY" if recent_direction > 0 else "SELL"
                confidence = min(0.45 + (ratio / self.spike_multiple) * 0.1, 0.7)
                signals.append(
                    Signal(
                        market_id=market.market_id,
                        token_id=token_id,
                        outcome=outcome,
                        side=side,
                        confidence=confidence,
                        target_price=current_price,
                        reasoning=f"volume/liquidity ratio {ratio:.1f}x (threshold {self.spike_multiple}x), "
                                  f"latest tick direction {'+' if recent_direction > 0 else '-'}",
                        source_agent=self.name,
                    )
                )
        return signals
