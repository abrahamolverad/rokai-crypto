# solana_meme_genie/solscan_listener.py

import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)

SOLSCAN_NEW_TOKENS_URL = "https://public-api.solscan.io/token/list"

async def fetch_new_tokens():
    """Fetches the latest created Solana tokens from Solscan."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(SOLSCAN_NEW_TOKENS_URL) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("data", [])
                else:
                    logger.error(f"Failed to fetch Solscan tokens. Status code: {resp.status}")
                    return []
    except Exception as e:
        logger.error(f"Error fetching new tokens from Solscan: {e}", exc_info=True)
        return []

async def monitor_new_tokens(callback_function, scan_interval=5):
    """Monitors Solscan for new tokens and calls callback when detected."""
    seen_tokens = set()

    while True:
        new_tokens = await fetch_new_tokens()

        for token in new_tokens:
            address = token.get("tokenAddress")
            symbol = token.get("symbol")

            if address and address not in seen_tokens:
                seen_tokens.add(address)
                logger.info(f"New Token Detected: {symbol} - {address}")
                await callback_function(token)

        await asyncio.sleep(scan_interval)
