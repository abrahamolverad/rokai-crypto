# solana_meme_genie/birdeye_scraper_playwright.py

import asyncio
import logging
import re
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

BIRDEYE_NEW_TOKENS_URL = "https://birdeye.so/new-tokens/solana"

async def fetch_new_tokens():
    """Uses Playwright to scrape the Birdeye new tokens page."""
    try:
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=True)
            page = await browser.new_page()
            await page.goto(BIRDEYE_NEW_TOKENS_URL)

            # Wait until the tokens appear
            await page.wait_for_selector("a[href^='/token/']", timeout=10000)

            html = await page.content()
            addresses = re.findall(r'/token/([A-Za-z0-9]{32,44})', html)
            addresses = list(set(addresses))  # Remove duplicates

            await browser.close()
            return addresses
    except Exception as e:
        logger.error(f"Error scraping Birdeye with Playwright: {e}", exc_info=True)
        return []

async def monitor_new_tokens(callback_function, scan_interval=5):
    """Monitors Birdeye for new tokens."""
    seen_tokens = set()

    while True:
        new_addresses = await fetch_new_tokens()

        for address in new_addresses:
            if address and address not in seen_tokens:
                seen_tokens.add(address)
                logger.info(f"New Token Detected: {address}")
                await callback_function({"address": address, "symbol": address[:4] + "..."})

        await asyncio.sleep(scan_interval)
