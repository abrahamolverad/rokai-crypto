# create_helius_webhook.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")

def create_webhook():
    url = f"https://api.helius.xyz/v0/webhooks?api-key={HELIUS_API_KEY}"

    data = {
        "webhookURL": "https://your-local-server.com/solana-mint",  # Placeholder for now (will adjust)
        "transactionTypes": ["INITIALIZE_MINT"],  # Listen to token mints
        "accountAddresses": [],  # Empty = listen to all addresses
        "webhookType": "enhanced",
        "encoding": "jsonParsed",
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("✅ Webhook created successfully!")
        print(response.json())
    else:
        print(f"❌ Error creating webhook: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    create_webhook()
