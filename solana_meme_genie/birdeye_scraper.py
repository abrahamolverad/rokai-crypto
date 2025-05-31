# solana_meme_genie/birdeye_scraper.py

import aiohttp
import asyncio
import logging
import re

logger = logging.getLogger(__name__)

BIRDEYE_NEW_TOKENS_PAGE = "https://birdeye.so/new-tokens/solana"

FAKE_HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.5",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
}

async def fetch_new_tokens():
    """Scrapes the Birdeye new tokens page for fresh token addresses."""
    try:
        async with aiohttp.ClientSession(headers=FAKE_HEADERS) as session:
            async with session.get(BIRDEYE_NEW_TOKENS_PAGE) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    addresses = re.findall(r'/token/([A-Za-z0-9]{32,44})', html)
                    addresses = list(set(addresses))  # remove duplicates
                    return addresses
                else:
                    logger.error(f"Failed to fetch Birdeye page. Status code: {resp.status}")
                    return []
    except Exception as e:
        logger.error(f"Error scraping Birdeye: {e}", exc_info=True)
        return []

async def monitor_new_tokens(callback_function, scan_interval=5):
    """Monitors Birdeye frontend for new tokens."""
    seen_tokens = set()

    while True:
        new_addresses = await fetch_new_tokens()

        for address in new_addresses:
            if address and address not in seen_tokens:
                seen_tokens.add(address)
                logger.info(f"New Token Detected: {address}")
                await callback_function({"address": address, "symbol": address[:4] + "..."})

        await asyncio.sleep(scan_interval)
