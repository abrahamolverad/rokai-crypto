# solana_meme_genie/solana_telegram.py

import aiohttp
import config_solana
import logging

logger = logging.getLogger(__name__)

async def send_solana_alert(message: str):
    try:
        telegram_url = f"https://api.telegram.org/bot{config_solana.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": config_solana.TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(telegram_url, json=payload) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to send Solana alert. HTTP {resp.status}")
                else:
                    logger.info(f"Sent Solana alert: {message}")
    except Exception as e:
        logger.error(f"Error sending Telegram alert: {e}", exc_info=True)
