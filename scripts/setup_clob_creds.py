"""Run this ONCE, locally, after funding your wallet with USDC on Polygon and
setting POLYGON_PRIVATE_KEY in your .env. It derives (or creates) your CLOB
API key/secret/passphrase from your wallet signature — these go into
CLOB_API_KEY / CLOB_API_SECRET / CLOB_API_PASSPHRASE (.env locally, or GitHub
Actions secrets for the scheduled workflow).

Usage:
    python scripts/setup_clob_creds.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()


def main():
    private_key = os.environ.get("POLYGON_PRIVATE_KEY", "")
    if not private_key or "yourprivatekeyhere" in private_key:
        print("Set POLYGON_PRIVATE_KEY in your .env first (a dedicated wallet, not your main one).")
        sys.exit(1)

    from py_clob_client_v2.client import ClobClient

    host = os.environ.get("CLOB_HOST", "https://clob.polymarket.com")
    chain_id = int(os.environ.get("CHAIN_ID", "137"))
    funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS") or None
    sig_type_raw = os.environ.get("POLYMARKET_SIGNATURE_TYPE")
    signature_type = int(sig_type_raw) if sig_type_raw else None

    client = ClobClient(
        host=host,
        key=private_key,
        chain_id=chain_id,
        funder=funder,
        signature_type=signature_type,
    )
    creds = client.create_or_derive_api_creds()

    print("\nAdd these to your .env / GitHub Actions secrets:\n")
    print(f"CLOB_API_KEY={creds.api_key}")
    print(f"CLOB_API_SECRET={creds.api_secret}")
    print(f"CLOB_API_PASSPHRASE={creds.api_passphrase}")
    print("\nDone. Keep these secret.")


if __name__ == "__main__":
    main()
