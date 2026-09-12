from __future__ import annotations

import json
import logging

from .base import BaseAgent, Signal

logger = logging.getLogger("polymarket_bot.agents.sentiment")

SYSTEM_PROMPT = """You are a probabilistic forecasting analyst reviewing a prediction \
market question. You will be given the market question, its description/resolution \
criteria, and its current implied prices (probabilities) for each outcome.

Your job: estimate your own probability for each outcome using your general \
knowledge, then compare it to the market's current price. Only flag a trade if \
your estimate differs from the market price by a meaningful margin AND you have \
reasonable grounds for that estimate (not pure speculation).

Respond with ONLY a JSON object, no markdown fences, no preamble:
{
  "outcome_estimates": [{"outcome": "<name>", "your_probability": 0.0-1.0}, ...],
  "confidence": 0.0-1.0,
  "reasoning": "<one or two sentences>",
  "recommend_trade": true/false
}

Set recommend_trade to false if you don't have enough information, if the market \
seems efficiently priced, or if your estimate is close (within ~5 points) to the \
market price. Be conservative — most markets are efficiently priced most of the time."""


class SentimentAgent(BaseAgent):
    """Uses the Claude API to form an independent probability estimate for a
    market and compares it against the current market price. This is the most
    token-expensive agent, so the caller (main.py) is responsible for capping
    how many markets get sent here per cycle (see config: max_sentiment_calls_per_cycle).
    """
    name = "sentiment"

    def __init__(self, cfg: dict, anthropic_client):
        super().__init__(cfg)
        self.client = anthropic_client
        self.model = cfg.get("model", "claude-sonnet-4-6")
        self.max_tokens = int(cfg.get("max_tokens", 600))

    def analyze(self, markets: list) -> list[Signal]:
        if not self.enabled or self.client is None:
            return []
        signals: list[Signal] = []
        for market in markets:
            try:
                estimate = self._estimate_market(market)
            except Exception as e:
                logger.warning("Sentiment analysis failed for market %s: %s", market.market_id, e)
                continue
            if estimate is None or not estimate.get("recommend_trade"):
                continue

            for est in estimate.get("outcome_estimates", []):
                outcome_name = est.get("outcome")
                your_prob = float(est.get("your_probability", 0))
                if outcome_name not in market.outcomes:
                    continue
                idx = market.outcomes.index(outcome_name)
                market_price = market.outcome_prices[idx]
                token_id = market.outcome_token_ids[idx]

                diff = your_prob - market_price
                if abs(diff) < 0.05:
                    continue

                side = "BUY" if diff > 0 else "SELL"
                signals.append(
                    Signal(
                        market_id=market.market_id,
                        token_id=token_id,
                        outcome=outcome_name,
                        side=side,
                        confidence=float(estimate.get("confidence", 0.5)),
                        target_price=market_price,
                        reasoning=f"Claude estimate {your_prob:.2f} vs market {market_price:.2f}: "
                                  f"{estimate.get('reasoning', '')}",
                        source_agent=self.name,
                    )
                )
        return signals

    def _estimate_market(self, market) -> dict | None:
        outcomes_str = ", ".join(
            f"{o} (current price {p:.2f})" for o, p in zip(market.outcomes, market.outcome_prices)
        )
        user_content = (
            f"Question: {market.question}\n"
            f"Description/resolution criteria: {market.description[:1500]}\n"
            f"End date: {market.end_date}\n"
            f"Outcomes and current market prices: {outcomes_str}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text")).strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse sentiment agent JSON response: %s", text[:200])
            return None
