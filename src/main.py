"""Entry point. Runs ONE full cycle: fetch markets -> run agents -> aggregate
-> risk-check -> (dry-run log OR place real orders) -> save state.

Designed to be invoked repeatedly by a scheduler (GitHub Actions cron) rather
than running as a long-lived process — each invocation is stateless except for
state.json, which is loaded/saved every run.
"""
from __future__ import annotations

import logging
import sys

from .agents.arbitrage_agent import ArbitrageAgent
from .agents.momentum_agent import MomentumAgent
from .agents.sentiment_agent import SentimentAgent
from .agents.volume_spike_agent import VolumeSpikeAgent
from .config import load_settings
from .coordinator import PortfolioCoordinator
from .polymarket_client import ClobExecutionClient, GammaClient
from .risk_manager import RiskManager
from .state_store import Position, load_state, save_state

logger = logging.getLogger("polymarket_bot.main")


def setup_logging(settings) -> None:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, settings.raw["logging"]["level"]),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(settings.log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def build_anthropic_client(settings):
    if not settings.anthropic_api_key:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def run_cycle(config_path: str | None = None) -> int:
    settings = load_settings(config_path)
    setup_logging(settings)
    logger.info("=== Starting cycle | dry_run=%s | kill_switch=%s ===", settings.dry_run, settings.kill_switch)

    state = load_state(settings.state_path)

    risk_manager = RiskManager(settings.risk, settings.bankroll_usdc)
    can_trade, reason = risk_manager.can_trade_at_all(state, settings.kill_switch)
    if not can_trade:
        logger.warning("Trading halted this cycle: %s", reason)
        save_state(settings.state_path, state)
        return 0

    gamma = GammaClient()
    scan_cfg = settings.market_scan
    markets = gamma.fetch_active_markets(limit=scan_cfg["max_markets_per_cycle"])

    min_vol = settings.risk["min_market_volume_usdc"]
    min_liq = settings.risk["min_market_liquidity_usdc"]
    markets = [m for m in markets if m.volume_24h_usdc >= min_vol and m.liquidity_usdc >= min_liq]
    logger.info("Fetched %d markets passing liquidity/volume filters", len(markets))

    if scan_cfg.get("categories"):
        markets = [m for m in markets if m.category in scan_cfg["categories"]]
    if scan_cfg.get("keywords"):
        kws = [k.lower() for k in scan_cfg["keywords"]]
        markets = [m for m in markets if any(k in m.question.lower() for k in kws)]

    agents_cfg = settings.agents_cfg
    all_signals = []

    momentum = MomentumAgent(agents_cfg["momentum"], gamma)
    all_signals += momentum.analyze(markets)

    volume_spike = VolumeSpikeAgent(agents_cfg["volume_spike"], gamma)
    all_signals += volume_spike.analyze(markets)

    arbitrage = ArbitrageAgent(agents_cfg["arbitrage"])
    all_signals += arbitrage.analyze(markets)

    # Sentiment agent is the expensive one — only send it the top N markets by volume
    # that don't already have an arbitrage signal (arbitrage is higher-confidence and free).
    if agents_cfg["sentiment"].get("enabled"):
        anthropic_client = build_anthropic_client(settings)
        sentiment = SentimentAgent(agents_cfg["sentiment"], anthropic_client)
        cap = scan_cfg.get("max_sentiment_calls_per_cycle", 8)
        top_markets = sorted(markets, key=lambda m: m.volume_24h_usdc, reverse=True)[:cap]
        all_signals += sentiment.analyze(top_markets)

    logger.info("Collected %d raw signals from %d agents", len(all_signals),
                sum(1 for a in [momentum, volume_spike, arbitrage] if a.enabled))

    coordinator = PortfolioCoordinator(risk_manager)
    decisions = coordinator.aggregate(all_signals)
    approved = coordinator.finalize_with_risk(decisions, state)

    logger.info("%d trade decision(s) approved after risk checks", len(approved))

    executor = None if settings.dry_run else ClobExecutionClient(settings)
    if not settings.dry_run:
        problems = settings.validate_secrets_for_live_trading()
        if problems:
            logger.error("Refusing to trade live — missing config: %s", "; ".join(problems))
            save_state(settings.state_path, state)
            return 1

    for d in approved:
        logger.info(
            "DECISION: %s %s @ ~%.3f size=$%.2f confidence=%s market=%s\n  reasoning: %s",
            d.side, d.outcome, d.price, d.size_usdc, d.confidence, d.market_id, d.reasoning,
        )
        order_id = None
        if settings.dry_run:
            logger.info("[DRY RUN] would place order but dry_run=true — no real order sent")
        else:
            try:
                resp = executor.place_limit_order(
                    token_id=d.token_id,
                    side=d.side,
                    price=d.price,
                    size_usdc=d.size_usdc,
                    order_type=settings.execution["order_type"],
                )
                order_id = resp.get("orderID") or resp.get("id")
            except Exception as e:
                logger.error("Order placement failed for market %s: %s", d.market_id, e)
                continue

        state.positions.append(
            Position(
                market_id=d.market_id,
                token_id=d.token_id,
                outcome=d.outcome,
                side=d.side,
                size_usdc=d.size_usdc,
                entry_price=d.price,
                opened_at=state.last_run_at or "",
                strategy_source=d.reasoning[:200],
                order_id=order_id,
            )
        )
        state.trade_log.append(
            {
                "market_id": d.market_id,
                "outcome": d.outcome,
                "side": d.side,
                "price": d.price,
                "size_usdc": d.size_usdc,
                "dry_run": settings.dry_run,
            }
        )

    save_state(settings.state_path, state)
    logger.info("=== Cycle complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(run_cycle())
