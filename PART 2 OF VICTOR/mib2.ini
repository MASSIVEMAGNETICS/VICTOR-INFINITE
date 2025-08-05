# ███╗   ███╗██╗██████╗     ██████╗  ██████╗ ████████╗
# ████╗ ████║██║██╔══██╗    ██╔══██╗██╔═══██╗╚══██╔══╝
# ██╔████╔██║██║██████╔╝    ██████╔╝██║   ██║   ██║
# ██║╚██╔╝██║██║██╔══██╗    ██╔══██╗██║   ██║   ██║
# ██║ ╚═╝ ██║██║██║  ██║    ██████╔╝╚██████╔╝   ██║
# ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝    ╚═════╝  ╚═════╝    ╚═╝
# Momentum Intraday Bot (MIB-2)
# v1.0 - For integration with Victor Watchdog
#
# TL;DR: Ride 5-min EMA crossovers—cut losers fast.

import ccxt.pro as ccxt  # Use the professional version for WebSockets
import asyncio
import pandas as pd
import os
import logging
from dotenv import load_dotenv

# --- Victor Watchdog & Logging Configuration ---
# Setup basic logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - MIB-2 - %(message)s',
    handlers=[
        logging.FileHandler("mib-2-log.txt"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file for security
load_dotenv()

# --- Core Configuration ---
# IMPORTANT: Do NOT hardcode keys. Use environment variables.
EXCHANGE_ID = 'binance' # Target exchange (e.g., 'binance', 'coinbasepro', 'kraken')
API_KEY = os.getenv('EXCHANGE_API_KEY')
SECRET_KEY = os.getenv('EXCHANGE_SECRET_KEY')
SYMBOL = 'BTC/USDT'     # Trading pair
TIMEFRAME = '5m'        # 5-minute candles
ORDER_AMOUNT = 0.001    # Amount of the base currency (BTC) to trade
SHORT_EMA_PERIOD = 5    # Short-term Exponential Moving Average
LONG_EMA_PERIOD = 20    # Long-term Exponential Moving Average

# --- Risk Management (Hooks for Victor Watchdog) ---
MAX_CONCURRENT_POSITIONS = 1
STOP_LOSS_PERCENT = 2.0 # 2% stop loss from entry price
TAKE_PROFIT_PERCENT = 5.0 # 5% take profit from entry price
ENABLE_RISK_THROTTLE = True # Master switch for Victor's risk control

# --- Global State ---
# This dictionary will hold the state of our positions.
# In a real-world scenario, this should be persisted in a database (e.g., Redis)
# to survive restarts.
position_state = {
    "in_position": False,
    "entry_price": 0.0,
    "position_size": 0.0,
    "last_order_id": None
}

class TradingBot:
    def __init__(self):
        """Initializes the bot and the exchange connection."""
        self.exchange = None
        self.ohlcv_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.initialize_exchange()

    def initialize_exchange(self):
        """Sets up the CCXT exchange instance with API credentials."""
        try:
            exchange_class = getattr(ccxt, EXCHANGE_ID)
            self.exchange = exchange_class({
                'apiKey': API_KEY,
                'secret': SECRET_KEY,
                'options': {
                    'defaultType': 'spot',
                },
            })
            # Use sandbox for testing if available
            if 'test' in self.exchange.urls:
                self.exchange.urls['api'] = self.exchange.urls['test']
                logging.warning("Exchange is in SANDBOX/TEST mode.")
            logging.info(f"Successfully initialized exchange: {EXCHANGE_ID}")
        except Exception as e:
            logging.error(f"Failed to initialize exchange: {e}")
            # VICTOR HOOK: Send critical alert on initialization failure
            exit()

    async def fetch_historical_data(self):
        """Fetches initial historical data to warm up the EMAs."""
        try:
            logging.info(f"Fetching historical data for {SYMBOL} to warm up indicators...")
            # We need at least `LONG_EMA_PERIOD` candles to start
            limit = LONG_EMA_PERIOD * 2
            ohlcv = await self.exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=limit)
            self.ohlcv_data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.ohlcv_data['timestamp'] = pd.to_datetime(self.ohlcv_data['timestamp'], unit='ms')
            logging.info(f"Successfully fetched {len(self.ohlcv_data)} initial candles.")
        except Exception as e:
            logging.error(f"Error fetching historical data: {e}")
            # VICTOR HOOK: Retry logic or alert
            await asyncio.sleep(60) # Wait before retrying
            await self.fetch_historical_data()

    def calculate_indicators(self):
        """Calculates short and long EMAs from the current OHLCV data."""
        if len(self.ohlcv_data) < LONG_EMA_PERIOD:
            return None, None # Not enough data yet

        # Use the 'close' price for EMA calculation
        close_prices = self.ohlcv_data['close']
        short_ema = close_prices.ewm(span=SHORT_EMA_PERIOD, adjust=False).mean().iloc[-1]
        long_ema = close_prices.ewm(span=LONG_EMA_PERIOD, adjust=False).mean().iloc[-1]
        return short_ema, long_ema

    async def check_risk_conditions(self, current_price):
        """
        Monitors open positions for stop-loss or take-profit triggers.
        This is the core of the risk throttle.
        """
        if not position_state["in_position"] or not ENABLE_RISK_THROTTLE:
            return

        entry_price = position_state["entry_price"]
        stop_loss_price = entry_price * (1 - STOP_LOSS_PERCENT / 100)
        take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENT / 100)

        # Check for stop-loss
        if current_price <= stop_loss_price:
            logging.warning(f"STOP-LOSS triggered at {current_price:.2f}. Selling position.")
            await self.execute_order('sell', ORDER_AMOUNT)
            # VICTOR HOOK: Log stop-loss event with position details

        # Check for take-profit
        elif current_price >= take_profit_price:
            logging.info(f"TAKE-PROFIT triggered at {current_price:.2f}. Selling position.")
            await self.execute_order('sell', ORDER_AMOUNT)
            # VICTOR HOOK: Log take-profit event with position details

    async def execute_order(self, side, amount):
        """
        Executes a market order and updates the global position state.
        `side` can be 'buy' or 'sell'.
        """
        if side == 'buy' and position_state["in_position"]:
            logging.warning("Buy signal received, but already in a position. Ignoring.")
            return
        if side == 'sell' and not position_state["in_position"]:
            logging.warning("Sell signal received, but not in a position. Ignoring.")
            return

        try:
            logging.info(f"Attempting to execute market {side} order for {amount} {SYMBOL.split('/')[0]}...")
            order = await self.exchange.create_market_order(SYMBOL, side, amount)
            logging.info(f"Order successfully executed: {order['id']}")

            # Update state based on order type
            if side == 'buy':
                position_state["in_position"] = True
                position_state["entry_price"] = order['price']
                position_state["position_size"] = order['amount']
                position_state["last_order_id"] = order['id']
            elif side == 'sell':
                # Reset state after selling
                position_state["in_position"] = False
                position_state["entry_price"] = 0.0
                position_state["position_size"] = 0.0
                # VICTOR HOOK: Calculate and log P&L for this trade
                # PnL = (order['price'] - entry_price) * amount
        except ccxt.InsufficientFunds as e:
            logging.error(f"Insufficient funds to execute {side} order: {e}")
            # VICTOR HOOK: Send critical alert about insufficient funds
        except Exception as e:
            logging.error(f"An error occurred during order execution: {e}")
            # VICTOR HOOK: Attempt to query order status or retry

    async def main_loop(self):
        """The core logic loop that watches for new candles and triggers trades."""
        await self.fetch_historical_data()

        while True:
            try:
                # Watch for the next candle
                candles = await self.exchange.watch_ohlcv(SYMBOL, TIMEFRAME)
                new_candle_df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Append new candle data, ensuring no duplicates
                # The last candle from fetch might be incomplete, so we replace it
                self.ohlcv_data = pd.concat([self.ohlcv_data.iloc[:-1], new_candle_df]).drop_duplicates(subset=['timestamp'], keep='last')
                self.ohlcv_data.reset_index(drop=True, inplace=True)

                # Get latest price and indicators
                latest_price = self.ohlcv_data['close'].iloc[-1]
                short_ema, long_ema = self.calculate_indicators()

                if short_ema is None or long_ema is None:
                    logging.info("Warming up indicators, not enough data yet...")
                    continue

                logging.info(f"Price: {latest_price:.2f} | EMA({SHORT_EMA_PERIOD}): {short_ema:.2f} | EMA({LONG_EMA_PERIOD}): {long_ema:.2f}")

                # --- Risk Check (runs every tick) ---
                await self.check_risk_conditions(latest_price)

                # --- Trading Logic (EMA Crossover) ---
                # Golden Cross: Short EMA crosses above Long EMA -> BUY signal
                if short_ema > long_ema and not position_state["in_position"]:
                    logging.info("Golden Cross detected! Firing BUY signal.")
                    await self.execute_order('buy', ORDER_AMOUNT)

                # Death Cross: Short EMA crosses below Long EMA -> SELL signal
                elif short_ema < long_ema and position_state["in_position"]:
                    logging.info("Death Cross detected! Firing SELL signal.")
                    await self.execute_order('sell', ORDER_AMOUNT)

            except ccxt.NetworkError as e:
                logging.warning(f"Network error: {e}. Reconnecting in 30s...")
                await asyncio.sleep(30)
            except Exception as e:
                logging.error(f"An unexpected error occurred in the main loop: {e}")
                # VICTOR HOOK: Send critical alert and attempt graceful shutdown/restart
                await asyncio.sleep(60) # Wait before restarting loop

async def run():
    """Entry point for the bot."""
    logging.info("Starting MIB-2 Trading Bot...")
    bot = TradingBot()
    await bot.main_loop()

if __name__ == "__main__":
    # To run this bot:
    # 1. Install dependencies: pip install "ccxt[pro]" pandas python-dotenv
    # 2. Create a .env file in the same directory with your keys:
    #    EXCHANGE_API_KEY=your_api_key_here
    #    EXCHANGE_SECRET_KEY=your_secret_key_here
    # 3. Run the script: python your_bot_file.py
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logging.info("Bot shutdown initiated by user.")
    finally:
        # Perform any cleanup here
        logging.info("MIB-2 has been shut down.")

