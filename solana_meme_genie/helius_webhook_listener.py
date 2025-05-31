# solana_meme_genie/helius_webhook_listener.py

from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route('/helius', methods=['POST'])
def helius_webhook():
    data = request.get_json(force=True)  # <-- force=True forces Flask to parse JSON correctly

    if data:
        try:
            token_mint_address = data["events"][0]["mint"]
            logger.info(f"New Token Mint Detected: {token_mint_address}")
            print(f"🔥 NEW TOKEN DETECTED: {token_mint_address}")
        except Exception as e:
            logger.error(f"Error processing webhook event: {e}")

    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(port=5000)
