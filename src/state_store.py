"""Persists bot state (open positions, daily P&L, kill-switch trips) to a JSON
file so it survives between stateless GitHub Actions runs. The workflow commits
this file back to the repo after every run — see .github/workflows/trade.yml.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Position:
    market_id: str
    token_id: str
    outcome: str
    side: str            # "BUY" or "SELL"
    size_usdc: float
    entry_price: float
    opened_at: str
    strategy_source: str
    order_id: str | None = None


@dataclass
class BotState:
    positions: list[Position] = field(default_factory=list)
    realized_pnl_today_usdc: float = 0.0
    trading_day: str = field(default_factory=lambda: date.today().isoformat())
    daily_loss_limit_hit: bool = False
    trade_log: list[dict[str, Any]] = field(default_factory=list)
    last_run_at: str | None = None

    def roll_day_if_needed(self) -> None:
        today = date.today().isoformat()
        if self.trading_day != today:
            self.trading_day = today
            self.realized_pnl_today_usdc = 0.0
            self.daily_loss_limit_hit = False

    def total_open_risk_usdc(self) -> float:
        return sum(p.size_usdc for p in self.positions)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BotState":
        positions = [Position(**p) for p in d.get("positions", [])]
        return cls(
            positions=positions,
            realized_pnl_today_usdc=d.get("realized_pnl_today_usdc", 0.0),
            trading_day=d.get("trading_day", date.today().isoformat()),
            daily_loss_limit_hit=d.get("daily_loss_limit_hit", False),
            trade_log=d.get("trade_log", []),
            last_run_at=d.get("last_run_at"),
        )


def load_state(path: Path) -> BotState:
    if not path.exists():
        return BotState()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    state = BotState.from_dict(raw)
    state.roll_day_if_needed()
    return state


def save_state(path: Path, state: BotState) -> None:
    state.last_run_at = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
