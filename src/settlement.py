"""Checks open positions against market resolution status. When a market has
resolved, computes realized P&L and folds it into current_bankroll_usdc — this
is what makes position sizing compound as the account grows or shrinks.

Polymarket binary markets resolve each outcome token to exactly $0 or $1.
"""
from __future__ import annotations

import logging

from .polymarket_client import GammaClient
from .state_store import BotState

logger = logging.getLogger("polymarket_bot.settlement")


def settle_resolved_positions(state: BotState, gamma: GammaClient) -> None:
    if not state.positions:
        return

    still_open = []
    for pos in state.positions:
        try:
            resp = gamma.session.get(
                "https://gamma-api.polymarket.com/markets",
                params={"id": pos.market_id},
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json()
        except Exception as e:
            logger.warning("Could not check resolution status for market %s: %s", pos.market_id, e)
            still_open.append(pos)
            continue

        if not results:
            still_open.append(pos)
            continue

        market = results[0]
        if not market.get("closed"):
            still_open.append(pos)
            continue

        # Market resolved. Find this position's outcome's final price (0 or 1).
        import json as _json
        outcomes = _json.loads(market.get("outcomes", "[]")) if isinstance(market.get("outcomes"), str) else market.get("outcomes", [])
        prices_raw = market.get("outcomePrices", "[]")
        prices = _json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        prices = [float(p) for p in prices]

        if pos.outcome not in outcomes:
            logger.warning("Resolved market %s missing outcome %s in results", pos.market_id, pos.outcome)
            still_open.append(pos)
            continue

        final_price = prices[outcomes.index(pos.outcome)]
        shares = pos.size_usdc / pos.entry_price if pos.entry_price > 0 else 0
        payout = shares * final_price if pos.side == "BUY" else shares * (pos.entry_price - final_price)
        pnl = payout - pos.size_usdc

        state.current_bankroll_usdc = round((state.current_bankroll_usdc or 0) + pnl, 4)
        state.realized_pnl_today_usdc = round(state.realized_pnl_today_usdc + pnl, 4)
        logger.info(
            "Settled position: market=%s outcome=%s side=%s pnl=%.2f USDC -> bankroll now %.2f USDC",
            pos.market_id, pos.outcome, pos.side, pnl, state.current_bankroll_usdc,
        )
        state.trade_log.append(
            {"market_id": pos.market_id, "outcome": pos.outcome, "event": "settled", "pnl_usdc": pnl}
        )

    state.positions = still_open
