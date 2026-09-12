"""Loads config.yaml and environment variables into a single settings object."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Settings:
    raw: dict[str, Any]

    # --- env secrets ---
    polygon_private_key: str = field(default_factory=lambda: os.environ.get("POLYGON_PRIVATE_KEY", ""))
    clob_api_key: str = field(default_factory=lambda: os.environ.get("CLOB_API_KEY", ""))
    clob_api_secret: str = field(default_factory=lambda: os.environ.get("CLOB_API_SECRET", ""))
    clob_api_passphrase: str = field(default_factory=lambda: os.environ.get("CLOB_API_PASSPHRASE", ""))
    clob_host: str = field(default_factory=lambda: os.environ.get("CLOB_HOST", "https://clob.polymarket.com"))
    chain_id: int = field(default_factory=lambda: int(os.environ.get("CHAIN_ID", "137")))
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    # Polymarket's "deposit wallet" / proxy address that actually holds funds
    # and is used as the order's maker — distinct from the signing EOA above
    # for most accounts (browser wallets included, per Polymarket's 2026
    # deposit-wallet migration). Find it on polymarket.com under your profile
    # / wallet settings. Leave blank to fall back to the signer's own address
    # (works only for accounts still on the old direct-EOA flow, if any).
    polymarket_funder_address: str = field(default_factory=lambda: os.environ.get("POLYMARKET_FUNDER_ADDRESS", ""))
    # 0 = EOA, 1 = POLY_PROXY (email/Magic-link wallets), 2 = POLY_GNOSIS_SAFE,
    # 3 = POLY_1271 (2026 deposit wallets). Leave unset to let the library default.
    polymarket_signature_type: int | None = field(
        default_factory=lambda: (
            int(os.environ["POLYMARKET_SIGNATURE_TYPE"])
            if os.environ.get("POLYMARKET_SIGNATURE_TYPE")
            else None
        )
    )

    @property
    def dry_run(self) -> bool:
        return bool(self.raw["mode"]["dry_run"])

    @property
    def kill_switch(self) -> bool:
        return bool(self.raw["mode"]["kill_switch"])

    @property
    def bankroll_usdc(self) -> float:
        return float(self.raw["capital"]["bankroll_usdc"])

    @property
    def risk(self) -> dict[str, Any]:
        return self.raw["risk"]

    @property
    def market_scan(self) -> dict[str, Any]:
        return self.raw["market_scan"]

    @property
    def agents_cfg(self) -> dict[str, Any]:
        return self.raw["agents"]

    @property
    def execution(self) -> dict[str, Any]:
        return self.raw["execution"]

    @property
    def state_path(self) -> Path:
        return REPO_ROOT / self.raw["state"]["file_path"]

    @property
    def log_path(self) -> Path:
        return REPO_ROOT / self.raw["logging"]["file_path"]

    def validate_secrets_for_live_trading(self) -> list[str]:
        """Returns a list of missing/problem items. Empty list = OK to trade live."""
        problems = []
        if not self.polygon_private_key or "yourprivatekeyhere" in self.polygon_private_key:
            problems.append("POLYGON_PRIVATE_KEY is missing or still a placeholder")
        if not self.clob_api_key:
            problems.append("CLOB_API_KEY is missing (run scripts/setup_clob_creds.py once)")
        if not self.anthropic_api_key and self.agents_cfg.get("sentiment", {}).get("enabled"):
            problems.append("ANTHROPIC_API_KEY is missing but sentiment agent is enabled")
        return problems


def load_settings(config_path: str | Path | None = None) -> Settings:
    path = Path(config_path) if config_path else REPO_ROOT / "config.yaml"
    raw = _load_yaml(path)
    return Settings(raw=raw)
