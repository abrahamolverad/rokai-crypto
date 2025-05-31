# solana_meme_genie/helius_listener.py

import asyncio
import aiohttp
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
HELIUS_URL = f"https://api.helius.xyz/v0/transactions/search?api-key={HELIUS_API_KEY}"

async def fetch_new_tokens(session):
    """Fetch recent token mint events from Helius."""
    try:
        payload = {
            "query": {
                "instructions": ["initializeMint"]
            },
            "limit": 20,
            "sortDirection": "desc"
        }
        async with session.post(HELIUS_URL, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("transactions", [])
            else:
                logger.error(f"Failed to fetch from Helius: {resp.status}")
                return None
    except Exception as e:
        logger.error(f"Error fetching from Helius: {e}", exc_info=True)
        return None

async def monitor_new_tokens(callback_function, scan_interval=5):
    """Monitors Solana blockchain for fresh token mints."""
    seen_mints = set()

    async with aiohttp.ClientSession() as session:
        while True:
            transactions = await fetch_new_tokens(session)

            if transactions:
                for tx in transactions:
                    try:
                        instructions = tx.get("parsedInstruction", [])
                        mint_address = instructions[0].get("info", {}).get("mint")
                        if mint_address and mint_address not in seen_mints:
                            seen_mints.add(mint_address)
                            logger.info(f"New Token Mint Detected: {mint_address}")
                            await callback_function({
                                "address": mint_address,
                                "symbol": mint_address[:4] + "..."
                            })
                    except Exception as e:
                        logger.error(f"Error parsing mint: {e}", exc_info=True)

            await asyncio.sleep(scan_interval)
