from __future__ import annotations

import logging
import time

from .base import BaseAgent, Signal

logger = logging.getLogger("polymarket_bot.agents.momentum")


class MomentumAgent(BaseAgent):
    """Flags outcomes whose price has moved sharply in the lookback window,
    on the theory that fresh information is being priced in and the move
    tends to continue briefly (momentum) rather than instantly mean-revert
    on these relatively illiquid markets.
    """
    name = "momentum"

    def __init__(self, cfg: dict, gamma_client):
        super().__init__(cfg)
        self.gamma_client = gamma_client
        self.lookback_minutes = int(cfg.get("lookback_minutes", 60))
        self.min_move_pct = float(cfg.get("min_price_move_pct", 4.0))

    def analyze(self, markets: list) -> list[Signal]:
        if not self.enabled:
            return []
        signals: list[Signal] = []
        for market in markets:
            for outcome, token_id, current_price in zip(
                market.outcomes, market.outcome_token_ids, market.outcome_prices
            ):
                try:
                    history = self.gamma_client.fetch_price_history(token_id, interval="1h")
                except Exception as e:
                    logger.debug("Price history fetch failed for %s: %s", token_id, e)
                    continue
                if len(history) < 2:
                    continue

                cutoff = time.time() - self.lookback_minutes * 60
                window = [pt for pt in history if pt.get("t", 0) >= cutoff]
                if len(window) < 2:
                    continue

                start_price = float(window[0]["p"])
                if start_price <= 0:
                    continue
                move_pct = (current_price - start_price) / start_price * 100

                if abs(move_pct) >= self.min_move_pct:
                    side = "BUY" if move_pct > 0 else "SELL"
                    confidence = min(0.5 + abs(move_pct) / 40.0, 0.85)
                    signals.append(
                        Signal(
                            market_id=market.market_id,
                            token_id=token_id,
                            outcome=outcome,
                            side=side,
                            confidence=confidence,
                            target_price=current_price,
                            reasoning=f"{move_pct:+.1f}% move over last {self.lookback_minutes}min "
                                      f"({start_price:.3f} -> {current_price:.3f})",
                            source_agent=self.name,
                        )
                    )
        return signals
