# solana_meme_genie/config_solana.py

import os
from dotenv import load_dotenv

load_dotenv()

# Solana Wallet
WALLET_PRIVATE_KEY = os.getenv("SOLANA_PYTHONGENERATEDWALLET1_PRIVATE_KEY")
WALLET_PUBLIC_KEY = os.getenv("SOLANA_PYTHONGENERATEDWALLET1_PUBLIC_KEY")
RPC_URL = os.getenv("RPC_URL_SOLANA")

# Telegram for Meme Coin Genie
TELEGRAM_BOT_TOKEN = os.getenv("MemeCon_Genie_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_USER_ID")

# Trading Settings
SOL_PER_TRADE = 0.3  # How much SOL to use per trade
MAX_ACTIVE_TRADES = 5
TAKE_PROFIT_MULTIPLIER = 2.0  # 2x TP
TRAILING_STOP_PERCENT = 20
EMERGENCY_STOP_LOSS = 30
MIN_LIQUIDITY_USD = 10000
MIN_LOCK_DAYS = 30

# Runtime Settings
SCAN_INTERVAL_SECONDS = 5
PRICE_CHECK_INTERVAL_SECONDS = 30

# Jupiter Settings
SLIPPAGE_BPS = 1000  # 10% slippage tolerance
