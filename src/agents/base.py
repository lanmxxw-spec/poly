from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Signal:
    """A single agent's opinion on a single market outcome."""
    market_id: str
    token_id: str
    outcome: str
    side: str            # "BUY" or "SELL"
    confidence: float     # 0.0 - 1.0
    target_price: float   # the price the agent thinks is fair / wants to trade at
    reasoning: str
    source_agent: str


class BaseAgent:
    name: str = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))

    def analyze(self, markets: list) -> list[Signal]:
        raise NotImplementedError
