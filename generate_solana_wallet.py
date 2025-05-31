# generate_solana_wallet.py

from solders.keypair import Keypair
import base64

def generate_wallet():
    keypair = Keypair()
    secret_key = keypair.to_bytes()
    public_key = keypair.pubkey()

    base64_secret_key = base64.b64encode(secret_key).decode()

    print("\n✅ Wallet generated successfully!\n")
    print(f"PRIVATE KEY (Base64 for .env):\n{base64_secret_key}\n")
    print(f"PUBLIC KEY (Wallet Address):\n{public_key}\n")

if __name__ == "__main__":
    generate_wallet()
