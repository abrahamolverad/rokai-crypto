# solana_meme_genie/solana_webhook_server.py

from flask import Flask, request
import asyncio
import logging
import os
from dotenv import load_dotenv
from jupiter_swapper import get_swap_quote, execute_swap  # already in your project
from solana_client import send_signed_transaction         # already in your project

load_dotenv()

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route("/solana-mint", methods=["POST"])
def handle_new_mint():
    data = request.get_json()
    print("🚀 New mint detected:", data)

    # Process mint event
    try:
        token_mint_address = data["transaction"]["message"]["accountKeys"][0]["pubkey"]
        logger.info(f"New Mint: {token_mint_address}")

        # Here you would call your sniper to swap it
        # Example:
        asyncio.run(trigger_sniper(token_mint_address))

    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)

    return "OK", 200

async def trigger_sniper(mint_address):
    logger.info(f"Attempting to snipe token: {mint_address}")
    quote = await get_swap_quote(mint_address)
    if quote:
        await execute_swap(quote)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
