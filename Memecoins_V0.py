#!/usr/bin/env python
"""
Memecoins_V0.py  — **original lightweight version**
Scans Birdeye’s legacy *new‑token* REST feed every 30 s and calls your filter /
buy logic.  **No API key required.**  This is the exact structure you were
running before any of the premium‑endpoint changes.

⚠️  Notes
---------
* Birdeye moved the endpoint in April 2025; `v1/token/new` now 301‑redirects to
  their docs page, so JSON parsing fails.  If you just want the old file back
  for reference / rollback, this is it — but be aware it will still throw the
  same exception you saw earlier (Expecting value: line 1 column 1).
* To make it functional again you’ll need either:
  1. The new premium endpoint (`/defi/v2/tokens/new_listing`) **with an API key**
     that has Premium access, **or**
  2. A different free source (e.g. Birdeye Trending, DexScreener, etc.).
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional

import requests

logging.basicConfig(
    format="%(asctime)s │ %(levelname)8s │ %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("memecoins_orig")

BIRDEYE_URL = "https://public-api.birdeye.so/v1/token/new"  # legacy endpoint
POLL_SECONDS = 30

# ───────────────────  data fetch  ───────────────────

def fetch_new_tokens(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch newest tokens from the old v1 endpoint (no key)."""
    params = {"limit": limit}
    try:
        r = requests.get(BIRDEYE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        log.info("Fetched %s new tokens", len(data))
        return data
    except Exception as e:
        log.error("Fetch failed: %s", e)
        return []

# ───────────────────  evaluation  ───────────────────

def evaluate_token(token: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Very simple filter (placeholder)."""
    if token.get("liquidity_usd", 0) < 20_000:
        return None
    return token

# ───────────────────  execution  ───────────────────

def execute_trade(token: Dict[str, Any]):
    symbol = token.get("symbol")
    address = token.get("address")
    log.info("🚀 BUY %s (%s)", symbol, address)
    # TODO: connect to trading backend

# ───────────────────  main loop  ───────────────────

def main():
    seen = set()
    while True:
        for token in fetch_new_tokens():
            addr = token.get("address")
            if addr in seen:
                continue
            seen.add(addr)
            approved = evaluate_token(token)
            if approved:
                execute_trade(approved)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user — exiting")
