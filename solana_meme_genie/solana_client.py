# solana_meme_genie/solana_client.py

import base64
import logging
from solana.rpc.async_api import AsyncClient
from solders.transaction import VersionedTransaction
from solders.keypair import Keypair
import config_solana

logger = logging.getLogger(__name__)

client = AsyncClient(config_solana.RPC_URL)

def load_keypair_from_private_key(private_key_str: str) -> Keypair:
    decoded = base64.b64decode(private_key_str)

    if len(decoded) == 64:
        return Keypair.from_bytes(decoded)
    elif len(decoded) == 66:
        return Keypair.from_bytes(decoded[:64])
    else:
        raise ValueError(f"Invalid private key length: {len(decoded)} bytes")

wallet = load_keypair_from_private_key(config_solana.WALLET_PRIVATE_KEY)
wallet_pubkey = wallet.pubkey()

logger.info(f"Loaded Solana Wallet: {wallet_pubkey}")

async def send_signed_transaction(encoded_txn: str) -> str:
    """
    Takes a base64-encoded unsigned transaction (from Jupiter),
    signs it with the wallet, and sends it to Solana blockchain.
    Returns transaction signature.
    """
    try:
        decoded_txn = base64.b64decode(encoded_txn)
        txn = VersionedTransaction.from_bytes(decoded_txn)

        # Sign the transaction with our wallet
        signed_txn = txn.sign([wallet])

        # Send the transaction
        txid = await client.send_raw_transaction(
            signed_txn.serialize(),
            opts={"skip_preflight": True, "preflight_commitment": "confirmed"}
        )

        logger.info(f"Transaction sent: {txid['result']}")
        return txid['result']
    except Exception as e:
        logger.error(f"Error sending transaction: {e}", exc_info=True)
        return None

async def close_connection():
    await client.close()
