# solana_meme_genie/solana_trading_loop.py

import asyncio
import logging
from jupiter_swapper import get_swap_quote, execute_swap
from solana_telegram import send_solana_alert
from helius_listener import monitor_new_tokens
import config_solana

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Constants === #
W_SOL_MINT = "So11111111111111111111111111111111111111112"  # Solana Wrapped SOL mint address

# === Active Trades === #
open_positions = {}

# === Callback for New Tokens === #
async def handle_new_token(token):
    address = token.get("address")
    symbol = token.get("symbol")
    liquidity = token.get("liquidity", 0)

    if liquidity < config_solana.MIN_LIQUIDITY_USD:
        logging.info(f"Skipping {symbol} due to low liquidity: ${liquidity}")
        return

    # Announce detection
    await send_solana_alert(f"🆕 New Token Detected: <b>{symbol}</b>")

    # === Step 1: Get Quote === #
    route = await get_swap_quote(W_SOL_MINT, address, config_solana.SOL_PER_TRADE)

    if not route:
        logging.warning(f"No valid swap route found for {symbol}. Skipping.")
        return

    # === Step 2: Execute Swap === #
    tx_signature = await execute_swap(route)

    if tx_signature:
        open_positions[address] = {
            "symbol": symbol,
            "buy_price": route['outAmount'] / route['inAmount'],
            "tx_signature": tx_signature,
            "highest_price": route['outAmount'] / route['inAmount']
        }
        await send_solana_alert(f"✅ Bought <b>{symbol}</b> successfully!")

# === Monitor Open Positions (basic TP/SL logic) === #
async def monitor_positions():
    while True:
        # (Simplified for now - just logs open positions)
        for token_addr, pos in open_positions.items():
            logging.info(f"[Open] {pos['symbol']} bought at {pos['buy_price']:.4f}")
        await asyncio.sleep(config_solana.PRICE_CHECK_INTERVAL_SECONDS)

# === Main Runner === #
async def main():
    task1 = asyncio.create_task(monitor_new_tokens(handle_new_token, scan_interval=config_solana.SCAN_INTERVAL_SECONDS))
    task2 = asyncio.create_task(monitor_positions())

    await asyncio.gather(task1, task2)

if __name__ == "__main__":
    asyncio.run(main())
