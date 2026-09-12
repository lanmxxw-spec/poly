"""Thin wrapper around Polymarket's public Gamma API (market/price data, no auth
needed) and the CLOB client (order placement, needs wallet + API creds).

Docs: https://docs.polymarket.com/
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Settings

logger = logging.getLogger("polymarket_bot.client")

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_PUBLIC_BASE = "https://clob.polymarket.com"


@dataclass
class MarketSnapshot:
    market_id: str
    question: str
    slug: str
    category: str
    active: bool
    closed: bool
    volume_24h_usdc: float
    liquidity_usdc: float
    outcomes: list[str]
    outcome_token_ids: list[str]
    outcome_prices: list[float]   # implied probabilities, should sum ~1.0 for binary markets
    end_date: str | None
    description: str = ""
    min_order_size_usdc: float = 1.0   # Polymarket sets this per-market; varies


class GammaClient:
    """Read-only market data — no auth required."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_active_markets(self, limit: int = 100, default_min_order: float = 1.0) -> list[MarketSnapshot]:
        resp = self.session.get(
            f"{GAMMA_BASE}/markets",
            params={"active": "true", "closed": "false", "limit": limit, "order": "volume24hr", "ascending": "false"},
            timeout=20,
        )
        resp.raise_for_status()
        raw_markets = resp.json()
        snapshots = []
        for m in raw_markets:
            try:
                snapshots.append(self._parse_market(m, default_min_order))
            except (KeyError, ValueError, TypeError) as e:
                logger.debug("Skipping unparseable market %s: %s", m.get("id"), e)
        return snapshots

    def fetch_market_min_order_size(self, condition_id: str, default: float = 1.0) -> float:
        """Best-effort lookup of a market's real minimum order size from the
        CLOB's public market endpoint. Falls back to `default` (Gamma's list
        endpoint doesn't reliably expose this, and per-market lookups cost an
        extra request, so this is only called for markets we're about to trade,
        not for every market scanned).
        """
        try:
            resp = self.session.get(f"{CLOB_PUBLIC_BASE}/markets/{condition_id}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for key in ("minimum_order_size", "minOrderSize", "min_order_size"):
                if key in data and data[key]:
                    return float(data[key])
        except Exception as e:
            logger.debug("Could not fetch min order size for %s, using default: %s", condition_id, e)
        return default

    def _parse_market(self, m: dict[str, Any], default_min_order: float = 1.0) -> MarketSnapshot:
        import json as _json

        outcomes = _json.loads(m.get("outcomes", "[]")) if isinstance(m.get("outcomes"), str) else m.get("outcomes", [])
        token_ids = (
            _json.loads(m.get("clobTokenIds", "[]"))
            if isinstance(m.get("clobTokenIds"), str)
            else m.get("clobTokenIds", [])
        )
        prices_raw = m.get("outcomePrices", "[]")
        prices = _json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        prices = [float(p) for p in prices]

        min_order = default_min_order
        for key in ("orderMinSize", "minimum_order_size", "minOrderSize"):
            if m.get(key):
                try:
                    min_order = float(m[key])
                    break
                except (TypeError, ValueError):
                    pass

        return MarketSnapshot(
            market_id=str(m["id"]),
            question=m.get("question", ""),
            slug=m.get("slug", ""),
            category=m.get("category", ""),
            active=bool(m.get("active", False)),
            closed=bool(m.get("closed", False)),
            volume_24h_usdc=float(m.get("volume24hr", 0) or 0),
            liquidity_usdc=float(m.get("liquidity", 0) or 0),
            outcomes=outcomes,
            outcome_token_ids=token_ids,
            outcome_prices=prices,
            end_date=m.get("endDate"),
            description=m.get("description", ""),
            min_order_size_usdc=min_order,
        )


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_price_history(self, token_id: str, interval: str = "1h") -> list[dict[str, Any]]:
        """Public endpoint, no auth needed. Returns list of {t: unix_ts, p: price}."""
        resp = self.session.get(
            f"{CLOB_PUBLIC_BASE}/prices-history",
            params={"market": token_id, "interval": interval, "fidelity": 10},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("history", [])


class ClobExecutionClient:
    """Wraps py-clob-client for authenticated order placement. Lazily imported
    so that dry-run / data-only usage doesn't require wallet creds to be set.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        creds = ApiCreds(
            api_key=self.settings.clob_api_key,
            api_secret=self.settings.clob_api_secret,
            api_passphrase=self.settings.clob_api_passphrase,
        )
        self._client = ClobClient(
            host=self.settings.clob_host,
            key=self.settings.polygon_private_key,
            chain_id=self.settings.chain_id,
            creds=creds,
        )
        return self._client

    def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size_usdc: float,
        order_type: str = "GTC",
    ) -> dict[str, Any]:
        """Places a real limit order. Only call this when settings.dry_run is False."""
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY, SELL

        client = self._ensure_client()
        side_const = BUY if side.upper() == "BUY" else SELL

        order_args = OrderArgs(
            token_id=token_id,
            price=round(price, 3),
            size=round(size_usdc / max(price, 0.01), 2),
            side=side_const,
        )
        signed_order = client.create_order(order_args)
        resp = client.post_order(signed_order, order_type)
        logger.info("Placed live order: token=%s side=%s price=%.3f size_usdc=%.2f -> %s",
                    token_id, side, price, size_usdc, resp)
        return resp

    def get_usdc_balance(self) -> float:
        client = self._ensure_client()
        balance = client.get_balance_allowance()
        return float(balance.get("balance", 0)) / 1_000_000  # USDC has 6 decimals
