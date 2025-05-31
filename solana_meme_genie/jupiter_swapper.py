# solana_meme_genie/jupiter_swapper.py

import aiohttp
import logging
import config_solana
from solana_client import send_signed_transaction
from solana_telegram import send_solana_alert

logger = logging.getLogger(__name__)

JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"

async def get_swap_quote(input_mint: str, output_mint: str, amount_in_sol: float):
    """Fetches the best route for a swap."""
    try:
        amount_in_lamports = int(amount_in_sol * 1_000_000_000)
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount_in_lamports,
            "slippageBps": config_solana.SLIPPAGE_BPS
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(JUPITER_QUOTE_URL, params=params) as resp:
                data = await resp.json()
                if data.get("data"):
                    return data["data"][0]  # Best route
                else:
                    logger.error("No route found in Jupiter quote response.")
                    return None
    except Exception as e:
        logger.error(f"Error fetching Jupiter quote: {e}", exc_info=True)
        return None

async def execute_swap(route: dict):
    """Executes a swap based on the route information."""
    try:
        payload = {
            "route": route,
            "userPublicKey": config_solana.WALLET_PUBLIC_KEY,
            "wrapUnwrapSOL": True,
            "feeAccount": None
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(JUPITER_SWAP_URL, json=payload) as resp:
                swap_data = await resp.json()
                encoded_tx = swap_data.get("swapTransaction")
                
                if not encoded_tx:
                    logger.error(f"No swap transaction returned: {swap_data}")
                    return None

                tx_signature = await send_signed_transaction(encoded_tx)
                if tx_signature:
                    await send_solana_alert(f"🚀 Swap Executed Successfully! TX: {tx_signature}")
                else:
                    await send_solana_alert(f"❌ Swap Failed.")
                
                return tx_signature
    except Exception as e:
        logger.error(f"Error executing Jupiter swap: {e}", exc_info=True)
        return None
