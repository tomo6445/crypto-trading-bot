def _collect_volume_data(self):
        """Collect volume data for pump and dump detection"""
        while self.running:
            for symbol in self.symbols:
                try:
                    # Get volume data from multiple timeframes
                    # 1m, 5m, 15m, 1h
                    timeframes = {
                        '1m': Client.KLINE_INTERVAL_1MINUTE,
                        '5m': Client.KLINE_INTERVAL_5MINUTE,
                        '15m': Client.KLINE_INTERVAL_15MINUTE,
                        '1h': Client.KLINE_INTERVAL_1HOUR
                    }
                    
                    volume_data = {}
                    for tf_name, tf_interval in timeframes.items():
                        # Get the most recent 100 candles for this timeframe
                        klines = self.client.get_klines(
                            symbol=symbol,
                            interval=tf_interval,
                            limit=100
                        )
                        
                        # Extract volume data
                        volumes = [float(k[5]) for k in klines]
                        
                        # Calculate volume metrics
                        avg_volume = np.mean(volumes[:-1])  # Average excluding most recent
                        current_volume = volumes[-1]
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                        
                        # Calculate volume momentum
                        volume_change_pct = []
                        for i in range(1, len(volumes)):
                            if volumes[i-1] > 0:
                                pct_change = (volumes[i] - volumes[i-1]) / volumes[i-1]
                                volume_change_pct.append(pct_change)
                            else:
                                volume_change_pct.append(0)
                        
                        # Calculate volume acceleration (change in volume change)
                        if len(volume_change_pct) > 1:
                            volume_accel = volume_change_pct[-1] - volume_change_pct[-2]
                        else:
                            volume_accel = 0
                        
                        # Store data for this timeframe
                        volume_data[tf_name] = {
                            'volumes': volumes,
                            'avg_volume': avg_volume,
                            'current_volume': current_volume,
                            'volume_ratio': volume_ratio,
                            'volume_change_pct': volume_change_pct,
                            'volume_accel': volume_accel,
                            'timestamp': time.time()
                        }
                    
                    # Store combined data for all timeframes
                    with self.lock:
                        self.pump_dump_data[symbol] = {
                            'volume_data': volume_data,
                            'timestamp': time.time()
                        }
                
                except Exception as e:
                    logger.error(f"Error collecting volume data for {symbol}: {e}")
            
            # Check every 30 seconds
            time.sleep(30)
    
    def _collect_social_sentiment(self):
        """Collect social media sentiment data for coins"""
        if not self.social_api_keys:
            logger.warning("No social API keys provided, skipping sentiment analysis")
            return
            
        # Initialize sentiment analyzer
        try:
            sentiment_analyzer = VaderSentimentAnalyzer()
        except:
            logger.warning("Failed to initialize VADER sentiment analyzer, falling back to NLTK")
            try:
                import nltk
                nltk.download('vader_lexicon', quiet=True)
                sentiment_analyzer = SentimentIntensityAnalyzer()
            except:
                logger.error("Failed to initialize sentiment analyzer, skipping sentiment analysis")
                return
        
        while self.running:
            # Extract the base symbols from trading pairs (e.g., BTC from BTCUSDT)
            base_symbols = [re.sub(r'USDT$|BTC$|ETH"""
Advanced Binance Trading Bot for Ultra-Short-Term Trading (30-60 min windows)
This bot implements multiple high-frequency trading strategies:
1. Order Book Imbalance Detection
2. Technical Indicator Momentum Scalping
3. Cross-Exchange Arbitrage
4. Statistical Arbitrage (Mean Reversion)
5. Pump and Dump Detection and Trading

Features:
- Real-time order book analysis
- ML-enhanced signal validation
- Adaptive position sizing
- Risk management with tight stop-losses
- Multi-strategy execution
- Social media sentiment analysis
- Volume anomaly detection
- Pattern recognition for pump and dump schemes

Requirements:
pip install python-binance pandas numpy scikit-learn tensorflow websocket-client ccxt nltk vaderSentiment requests tweepy praw
"""

import os
import time
import numpy as np
import pandas as pd
import threading
import json
import logging
import datetime
import warnings
import math
import csv
import re
import requests
import tweepy
import praw
from collections import defaultdict, Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentimentAnalyzer
from binance.client import Client
from binance.websockets import BinanceSocketManager
from binance.enums import *
import ccxt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

class BinanceHFTBot:
    def __init__(self, api_key, api_secret, target_gain=0.03, timeframe_minutes=60, 
                 base_trade_amount=100, max_open_trades=3, use_ml=True, 
                 exchanges=None, symbols=None, 
                 social_api_keys=None, enable_pump_dump_detection=True):
        """
        Initialize the high-frequency trading bot
        
        Parameters:
        - api_key: Binance API key
        - api_secret: Binance API secret
        - target_gain: Target gain percentage (default 3%)
        - timeframe_minutes: Maximum trade duration (default 60 minutes)
        - base_trade_amount: Base amount to trade in USDT
        - max_open_trades: Maximum number of concurrent trades
        - use_ml: Whether to use machine learning for signal validation
        - exchanges: List of additional exchanges for arbitrage (ccxt format)
        - symbols: List of trading pairs to monitor (default: top 5 by volume)
        - social_api_keys: API keys for social media platforms
        - enable_pump_dump_detection: Whether to enable pump and dump detection
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = Client(api_key, api_secret)
        self.bsm = None
        self.start_time = time.time()  # Record when the bot starts
        
        # Bot settings
        self.target_gain = target_gain
        self.timeframe_minutes = timeframe_minutes
        self.base_trade_amount = base_trade_amount
        self.max_open_trades = max_open_trades
        self.use_ml = use_ml
        self.enable_pump_dump_detection = enable_pump_dump_detection
        
        # State variables
        self.active_trades = {}
        self.order_book_data = {}
        self.technical_data = {}
        self.exchange_price_data = {}
        self.pump_dump_data = {}
        self.social_sentiment_data = {}
        self.running = False
        self.lock = threading.Lock()
        
        # Social API keys
        self.social_api_keys = social_api_keys or {}
        
        # Setup additional exchanges for arbitrage
        self.exchanges = {}
        if exchanges:
            for exchange_id, credentials in exchanges.items():
                exchange_class = getattr(ccxt, exchange_id)
                self.exchanges[exchange_id] = exchange_class({
                    'apiKey': credentials['api_key'],
                    'secret': credentials['api_secret'],
                    'enableRateLimit': True
                })
        
        # Add Binance to exchanges dict
        self.exchanges['binance'] = self.client
        
        # Determine symbols to trade
        if symbols:
            self.symbols = symbols
        else:
            # Get top 5 USDT pairs by volume
            exchange_info = self.client.get_exchange_info()
            tickers = self.client.get_ticker()
            usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
            usdt_pairs.sort(key=lambda x: float(x['volume']), reverse=True)
            self.symbols = [p['symbol'] for p in usdt_pairs[:5]]
        
        logger.info(f"Bot initialized with symbols: {self.symbols}")
        
        # ML models
        self.ml_models = {}
        if self.use_ml:
            self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for each symbol and strategy"""
        for symbol in self.symbols:
            self.ml_models[symbol] = {
                'order_book': None,
                'technical': None,
                'combined': None
            }
            
            # Load historical data for training
            try:
                # Get last 1000 1-minute klines for initial training
                klines = self.client.get_historical_klines(
                    symbol, Client.KLINE_INTERVAL_1MINUTE, "1000 minutes ago UTC"
                )
                
                if len(klines) > 100:  # Ensure we have enough data
                    df = self._prepare_training_data(symbol, klines)
                    self._train_models(symbol, df)
                    logger.info(f"ML models trained for {symbol}")
                else:
                    logger.warning(f"Not enough data to train ML models for {symbol}")
            except Exception as e:
                logger.error(f"Error training ML models for {symbol}: {e}")
    
    def _prepare_training_data(self, symbol, klines):
        """Prepare historical data for ML training"""
        # Convert klines to DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert string values to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Calculate features (similar to what we'll use in real-time)
        df['returns'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Technical indicators
        # SMA
        df['sma5'] = df['close'].rolling(5).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma_ratio'] = df['sma5'] / df['sma20']
        
        # Bollinger Bands
        df['std20'] = df['close'].rolling(20).std()
        df['upper_band'] = df['sma20'] + (df['std20'] * 2)
        df['lower_band'] = df['sma20'] - (df['std20'] * 2)
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['sma20']
        df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
        
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Target: Will price increase by target_gain within next 60 mins?
        future_prices = df['close'].shift(-60)  # Look 60 periods ahead (60 mins)
        df['target'] = (future_prices / df['close'] - 1 > self.target_gain).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def _train_models(self, symbol, df):
        """Train ML models using prepared data"""
        # Split data for training
        X = df[[
            'sma_ratio', 'bb_width', 'bb_position', 'rsi', 
            'macd', 'macd_hist', 'volume_change'
        ]]
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Store model and scaler
        self.ml_models[symbol]['technical'] = {
            'model': rf_model,
            'scaler': scaler,
            'features': X.columns.tolist()
        }
        
        # Also train a simpler GradientBoosting model for order book data
        # (features will be constructed in real-time)
        gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        # This is a placeholder - we'll train with actual order book data once collected
        self.ml_models[symbol]['order_book'] = {
            'model': gb_model,
            'features': ['imbalance', 'spread_pct', 'depth_ratio']
        }
        
        logger.info(f"Models trained for {symbol} with accuracy: {rf_model.score(X_test_scaled, y_test):.4f}")

    def start(self):
        """Start the trading bot"""
        if self.running:
            logger.warning("Bot already running")
            return
        
        self.running = True
        
        # Initialize Socket Manager
        self.bsm = BinanceSocketManager(self.client)
        
        # Start data collection threads
        threading.Thread(target=self._collect_order_book_data, daemon=True).start()
        threading.Thread(target=self._collect_technical_data, daemon=True).start()
        
        # If using arbitrage, start exchange data collection
        if len(self.exchanges) > 1:
            threading.Thread(target=self._collect_exchange_data, daemon=True).start()
        
        # If pump and dump detection is enabled, start related threads
        if self.enable_pump_dump_detection:
            threading.Thread(target=self._collect_volume_data, daemon=True).start()
            threading.Thread(target=self._collect_social_sentiment, daemon=True).start()
            threading.Thread(target=self._run_pump_dump_detection, daemon=True).start()
        
        # Start the main trading loop
        threading.Thread(target=self._trading_loop, daemon=True).start()
        
        # Start the trade monitoring loop
        threading.Thread(target=self._monitor_trades, daemon=True).start()
        
        logger.info("Bot started successfully")
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        if self.bsm:
            self.bsm.close()
        logger.info("Bot stopped")
    
    def _collect_order_book_data(self):
        """Collect real-time order book data"""
        connections = {}
        
        for symbol in self.symbols:
            # Initialize order book data structure
            self.order_book_data[symbol] = {
                'bids': {},
                'asks': {},
                'timestamp': 0,
                'computed': {
                    'imbalance': 0,
                    'spread': 0,
                    'spread_pct': 0,
                    'depth_ratio': 0,
                    'top_levels': []
                }
            }
            
            # Start order book web socket
            connections[symbol] = self.bsm.start_depth_socket(
                symbol, self._process_depth_message, depth=BinanceSocketManager.DEPTH_5
            )
        
        # Start socket manager
        self.bsm.start()
        
        logger.info(f"Started collecting order book data for {len(self.symbols)} symbols")
        
        # Keep the thread alive
        while self.running:
            time.sleep(1)
    
    def _process_depth_message(self, msg):
        """Process depth (order book) messages from websocket"""
        if 'e' not in msg or msg['e'] != 'depthUpdate':
            return
        
        symbol = msg['s']
        
        with self.lock:
            order_book = self.order_book_data.get(symbol)
            if not order_book:
                return
            
            # Update bids
            for bid in msg['b']:
                price, quantity = float(bid[0]), float(bid[1])
                if quantity == 0:
                    if price in order_book['bids']:
                        del order_book['bids'][price]
                else:
                    order_book['bids'][price] = quantity
            
            # Update asks
            for ask in msg['a']:
                price, quantity = float(ask[0]), float(ask[1])
                if quantity == 0:
                    if price in order_book['asks']:
                        del order_book['asks'][price]
                else:
                    order_book['asks'][price] = quantity
            
            # Update timestamp
            order_book['timestamp'] = msg['E']
            
            # Compute derived metrics
            self._compute_order_book_metrics(symbol)
    
    def _compute_order_book_metrics(self, symbol):
        """Compute derived metrics from order book data"""
        order_book = self.order_book_data[symbol]
        
        # Get sorted bids and asks
        bids = sorted(order_book['bids'].items(), key=lambda x: x[0], reverse=True)
        asks = sorted(order_book['asks'].items(), key=lambda x: x[0])
        
        if not bids or not asks:
            return
        
        # Get best bid and ask
        best_bid, best_bid_qty = bids[0]
        best_ask, best_ask_qty = asks[0]
        
        # Calculate spread
        spread = best_ask - best_bid
        spread_pct = spread / best_bid
        
        # Calculate imbalance (using top 5 levels or all available if fewer)
        bid_vol = sum(qty for _, qty in bids[:5])
        ask_vol = sum(qty for _, qty in asks[:5])
        
        # Order book imbalance formula from the paper
        if bid_vol + ask_vol > 0:
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        else:
            imbalance = 0
        
        # Calculate depth ratio (volume at best bid vs best ask)
        if best_ask_qty > 0:
            depth_ratio = best_bid_qty / best_ask_qty
        else:
            depth_ratio = 1
        
        # Store top 5 price levels for visualization/analysis
        top_levels = {
            'bids': bids[:5],
            'asks': asks[:5]
        }
        
        # Update computed metrics
        order_book['computed'] = {
            'imbalance': imbalance,
            'spread': spread,
            'spread_pct': spread_pct,
            'depth_ratio': depth_ratio,
            'top_levels': top_levels,
            'last_update': time.time()
        }
    
    def _collect_technical_data(self):
        """Collect and compute technical indicators from klines data"""
        while self.running:
            for symbol in self.symbols:
                try:
                    # Get the most recent 100 1-minute candles
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=Client.KLINE_INTERVAL_1MINUTE,
                        limit=100
                    )
                    
                    # Convert to dataframe
                    df = pd.DataFrame(klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert string values to numeric
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Calculate indicators
                    # SMA
                    df['sma5'] = df['close'].rolling(5).mean()
                    df['sma20'] = df['close'].rolling(20).mean()
                    df['sma_ratio'] = df['sma5'] / df['sma20']
                    
                    # Bollinger Bands
                    df['sma20'] = df['close'].rolling(20).mean()
                    df['std20'] = df['close'].rolling(20).std()
                    df['upper_band'] = df['sma20'] + (df['std20'] * 2)
                    df['lower_band'] = df['sma20'] - (df['std20'] * 2)
                    df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['sma20']
                    df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
                    
                    # RSI (14)
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    
                    # MACD
                    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
                    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
                    df['macd'] = df['ema12'] - df['ema26']
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
                    
                    # Volume analysis
                    df['volume_sma20'] = df['volume'].rolling(20).mean()
                    df['volume_ratio'] = df['volume'] / df['volume_sma20']
                    
                    # Price momentum
                    df['price_change_1m'] = df['close'].pct_change(1)
                    df['price_change_5m'] = df['close'].pct_change(5)
                    
                    # Store most recent data
                    with self.lock:
                        # Extract the most recent row with complete indicators
                        latest_data = df.iloc[-1].to_dict()
                        self.technical_data[symbol] = {
                            'data': latest_data,
                            'timestamp': time.time(),
                            'dataframe': df.tail(20)  # Keep recent history for analysis
                        }
                
                except Exception as e:
                    logger.error(f"Error collecting technical data for {symbol}: {e}")
            
            # Update every 30 seconds
            time.sleep(30)
    
    def _collect_exchange_data(self):
        """Collect price data from multiple exchanges for arbitrage"""
        while self.running:
            for symbol in self.symbols:
                symbol_data = {}
                
                # For each exchange, get ticker data
                for exchange_id, exchange in self.exchanges.items():
                    try:
                        if exchange_id == 'binance':
                            # For Binance, we use the Client
                            ticker = self.client.get_symbol_ticker(symbol=symbol)
                            price = float(ticker['price'])
                        else:
                            # For other exchanges, we use ccxt
                            # May need to standardize the symbol format for different exchanges
                            standardized_symbol = symbol.replace('USDT', '/USDT')
                            ticker = exchange.fetch_ticker(standardized_symbol)
                            price = ticker['last']
                        
                        symbol_data[exchange_id] = {
                            'price': price,
                            'timestamp': time.time()
                        }
                        
                    except Exception as e:
                        logger.error(f"Error fetching {symbol} price from {exchange_id}: {e}")
                
                # Store the exchange data
                with self.lock:
                    self.exchange_price_data[symbol] = symbol_data
            
            # Check for arbitrage opportunities every 5 seconds
            time.sleep(5)
    
    def _trading_loop(self):
        """Main trading loop that evaluates strategies and executes trades"""
        while self.running:
            # Skip if we're at max open trades
            if len(self.active_trades) >= self.max_open_trades:
                time.sleep(5)
                continue
            
            # Analyze each symbol
            for symbol in self.symbols:
                # Skip if we already have an active trade for this symbol
                if symbol in self.active_trades:
                    continue
                
                try:
                    # Get signals from different strategies
                    order_book_signal = self._get_order_book_signal(symbol)
                    technical_signal = self._get_technical_signal(symbol)
                    arbitrage_signal = self._get_arbitrage_signal(symbol)
                    
                    # Get pump and dump signal if enabled
                    pump_dump_signal = self._get_pump_dump_signal(symbol) if self.enable_pump_dump_detection else {'signal': 'neutral', 'confidence': 0, 'data': {}}
                    
                    # Combine signals and make decision
                    decision, confidence, strategy = self._make_trading_decision(
                        symbol, order_book_signal, technical_signal, arbitrage_signal, pump_dump_signal
                    )
                    
                    # Execute trade if decision is positive
                    if decision == 'buy' and confidence > 0.65:  # Confidence threshold
                        self._execute_trade(symbol, 'buy', confidence, strategy)
                
                except Exception as e:
                    logger.error(f"Error in trading loop for {symbol}: {e}")
            
            # Check every 5 seconds
            time.sleep(5)
    
    def _get_order_book_signal(self, symbol):
        """Generate trading signal based on order book imbalance"""
        with self.lock:
            if symbol not in self.order_book_data:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            ob_data = self.order_book_data[symbol]
            
            # Check if data is fresh (less than 5 seconds old)
            if 'computed' not in ob_data or time.time() - ob_data['computed'].get('last_update', 0) > 5:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            computed = ob_data['computed']
            imbalance = computed['imbalance']
            
            # Prepare feature vector for ML if we're using it
            if self.use_ml and symbol in self.ml_models and self.ml_models[symbol]['order_book']:
                features = [
                    imbalance,
                    computed['spread_pct'],
                    computed['depth_ratio']
                ]
                # We would need to have trained this model with actual data
                # This is just a placeholder for now
                # model = self.ml_models[symbol]['order_book']['model']
                # confidence = model.predict_proba([features])[0][1]  # Probability of class 1
                
                # For now, use a rule-based approach
                if imbalance > 0.3:  # Strong buy imbalance
                    signal = 'buy'
                    confidence = min(0.5 + imbalance/2, 0.9)  # Map imbalance to confidence
                elif imbalance < -0.3:  # Strong sell imbalance
                    signal = 'sell'
                    confidence = min(0.5 + abs(imbalance)/2, 0.9)
                else:
                    signal = 'neutral'
                    confidence = 0.5
            else:
                # Rule-based approach without ML
                if imbalance > 0.3:  # Strong buy imbalance
                    signal = 'buy'
                    confidence = min(0.5 + imbalance/2, 0.9)
                elif imbalance < -0.3:  # Strong sell imbalance
                    signal = 'sell'
                    confidence = min(0.5 + abs(imbalance)/2, 0.9)
                else:
                    signal = 'neutral'
                    confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': confidence,
                'data': {
                    'imbalance': imbalance,
                    'spread_pct': computed['spread_pct'],
                    'depth_ratio': computed['depth_ratio']
                }
            }
    
    def _get_technical_signal(self, symbol):
        """Generate trading signal based on technical indicators"""
        with self.lock:
            if symbol not in self.technical_data:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            tech_data = self.technical_data[symbol]
            data = tech_data['data']
            
            # Check if data is fresh (less than 2 minutes old)
            if time.time() - tech_data.get('timestamp', 0) > 120:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            # Use ML model if available
            if (self.use_ml and symbol in self.ml_models and 
                self.ml_models[symbol]['technical'] and 
                'model' in self.ml_models[symbol]['technical']):
                
                model_info = self.ml_models[symbol]['technical']
                model = model_info['model']
                scaler = model_info['scaler']
                features = model_info['features']
                
                # Extract feature values
                feature_values = []
                for feature in features:
                    feature_values.append(data.get(feature, 0))
                
                # Scale features
                scaled_features = scaler.transform([feature_values])
                
                # Get prediction and probability
                prediction = model.predict(scaled_features)[0]
                probabilities = model.predict_proba(scaled_features)[0]
                
                if prediction == 1:  # Buy signal
                    signal = 'buy'
                    confidence = probabilities[1]
                else:
                    signal = 'sell' if probabilities[0] > 0.7 else 'neutral'
                    confidence = probabilities[0] if signal == 'sell' else 0.5
            
            else:
                # Rule-based approach
                # Check for bullish conditions
                bullish_conditions = [
                    data.get('sma_ratio', 1) > 1.005,  # Fast SMA above slow SMA
                    data.get('macd', 0) > data.get('macd_signal', 0),  # MACD above signal
                    data.get('rsi', 50) > 50 and data.get('rsi', 50) < 70,  # RSI in bullish zone but not overbought
                    data.get('bb_position', 0.5) > 0.5 and data.get('bb_position', 0.5) < 0.9,  # Price in upper half of BB but not at extreme
                    data.get('volume_ratio', 1) > 1.2  # Higher than average volume
                ]
                
                # Check for bearish conditions
                bearish_conditions = [
                    data.get('sma_ratio', 1) < 0.995,  # Fast SMA below slow SMA
                    data.get('macd', 0) < data.get('macd_signal', 0),  # MACD below signal
                    data.get('rsi', 50) < 50 and data.get('rsi', 50) > 30,  # RSI in bearish zone but not oversold
                    data.get('bb_position', 0.5) < 0.5 and data.get('bb_position', 0.5) > 0.1,  # Price in lower half of BB but not at extreme
                    data.get('volume_ratio', 1) > 1.2  # Higher than average volume
                ]
                
                bullish_count = sum(bullish_conditions)
                bearish_count = sum(bearish_conditions)
                
                # Determine signal and confidence
                if bullish_count >= 3 and bullish_count > bearish_count + 1:
                    signal = 'buy'
                    confidence = 0.5 + (bullish_count / 10)
                elif bearish_count >= 3 and bearish_count > bullish_count + 1:
                    signal = 'sell'
                    confidence = 0.5 + (bearish_count / 10)
                else:
                    signal = 'neutral'
                    confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': confidence,
                'data': {
                    'sma_ratio': data.get('sma_ratio', 1),
                    'macd': data.get('macd', 0),
                    'rsi': data.get('rsi', 50),
                    'bb_position': data.get('bb_position', 0.5),
                    'volume_ratio': data.get('volume_ratio', 1)
                }
            }
    
    def _get_arbitrage_signal(self, symbol):
        """Generate trading signal based on arbitrage opportunities"""
        with self.lock:
            if symbol not in self.exchange_price_data or len(self.exchange_price_data[symbol]) < 2:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            symbol_data = self.exchange_price_data[symbol]
            
            # Check all prices are recent (less than 30 seconds old)
            for exchange, data in symbol_data.items():
                if time.time() - data.get('timestamp', 0) > 30:
                    return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            # Find min and max prices across exchanges
            prices = [(exchange, data['price']) for exchange, data in symbol_data.items()]
            min_exchange, min_price = min(prices, key=lambda x: x[1])
            max_exchange, max_price = max(prices, key=lambda x: x[1])
            
            # Calculate price difference percentage
            price_diff_pct = (max_price - min_price) / min_price
            
            # Factor in trading fees
            # Assume 0.1% fee per trade (0.2% round trip)
            arbitrage_threshold = 0.002  # 0.2% minimum profitable arbitrage
            
            # Determine signal and confidence
            if price_diff_pct > arbitrage_threshold:
                # Potential arbitrage opportunity
                signal = 'buy'  # Buy on cheaper exchange
                # Scale confidence based on how much the spread exceeds our threshold
                confidence = min(0.5 + (price_diff_pct - arbitrage_threshold) * 50, 0.95)
                
                # Store which exchange to buy on and which to sell on
                buy_exchange = min_exchange
                sell_exchange = max_exchange
            else:
                signal = 'neutral'
                confidence = 0.5
                buy_exchange = None
                sell_exchange = None
            
            return {
                'signal': signal,
                'confidence': confidence,
                'data': {
                    'price_diff_pct': price_diff_pct,
                    'buy_exchange': buy_exchange,
                    'sell_exchange': sell_exchange,
                    'min_price': min_price,
                    'max_price': max_price
                }
            }, '', symbol) for symbol in self.symbols]
            base_symbols = [s for s in base_symbols if s]  # Remove empty strings
            
            all_sentiments = {}
            
            # Collect Twitter data if API keys available
            if 'twitter' in self.social_api_keys:
                try:
                    twitter_sentiments = self._get_twitter_sentiment(base_symbols, sentiment_analyzer)
                    all_sentiments['twitter'] = twitter_sentiments
                except Exception as e:
                    logger.error(f"Error collecting Twitter sentiment: {e}")
            
            # Collect Reddit data if API keys available
            if 'reddit' in self.social_api_keys:
                try:
                    reddit_sentiments = self._get_reddit_sentiment(base_symbols, sentiment_analyzer)
                    all_sentiments['reddit'] = reddit_sentiments
                except Exception as e:
                    logger.error(f"Error collecting Reddit sentiment: {e}")
            
            # Process and store combined sentiment data
            for symbol in self.symbols:
                base_symbol = re.sub(r'USDT$|BTC$|ETH"""
Advanced Binance Trading Bot for Ultra-Short-Term Trading (30-60 min windows)
This bot implements multiple high-frequency trading strategies:
1. Order Book Imbalance Detection
2. Technical Indicator Momentum Scalping
3. Cross-Exchange Arbitrage
4. Statistical Arbitrage (Mean Reversion)
5. Pump and Dump Detection and Trading

Features:
- Real-time order book analysis
- ML-enhanced signal validation
- Adaptive position sizing
- Risk management with tight stop-losses
- Multi-strategy execution
- Social media sentiment analysis
- Volume anomaly detection
- Pattern recognition for pump and dump schemes

Requirements:
pip install python-binance pandas numpy scikit-learn tensorflow websocket-client ccxt nltk vaderSentiment requests tweepy praw
"""

import os
import time
import numpy as np
import pandas as pd
import threading
import json
import logging
import datetime
import warnings
import math
import csv
import re
import requests
import tweepy
import praw
from collections import defaultdict, Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentimentAnalyzer
from binance.client import Client
from binance.websockets import BinanceSocketManager
from binance.enums import *
import ccxt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

class BinanceHFTBot:
    def __init__(self, api_key, api_secret, target_gain=0.03, timeframe_minutes=60, 
                 base_trade_amount=100, max_open_trades=3, use_ml=True, 
                 exchanges=None, symbols=None, 
                 social_api_keys=None, enable_pump_dump_detection=True):
        """
        Initialize the high-frequency trading bot
        
        Parameters:
        - api_key: Binance API key
        - api_secret: Binance API secret
        - target_gain: Target gain percentage (default 3%)
        - timeframe_minutes: Maximum trade duration (default 60 minutes)
        - base_trade_amount: Base amount to trade in USDT
        - max_open_trades: Maximum number of concurrent trades
        - use_ml: Whether to use machine learning for signal validation
        - exchanges: List of additional exchanges for arbitrage (ccxt format)
        - symbols: List of trading pairs to monitor (default: top 5 by volume)
        - social_api_keys: API keys for social media platforms
        - enable_pump_dump_detection: Whether to enable pump and dump detection
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = Client(api_key, api_secret)
        self.bsm = None
        self.start_time = time.time()  # Record when the bot starts
        
        # Bot settings
        self.target_gain = target_gain
        self.timeframe_minutes = timeframe_minutes
        self.base_trade_amount = base_trade_amount
        self.max_open_trades = max_open_trades
        self.use_ml = use_ml
        self.enable_pump_dump_detection = enable_pump_dump_detection
        
        # State variables
        self.active_trades = {}
        self.order_book_data = {}
        self.technical_data = {}
        self.exchange_price_data = {}
        self.pump_dump_data = {}
        self.social_sentiment_data = {}
        self.running = False
        self.lock = threading.Lock()
        
        # Social API keys
        self.social_api_keys = social_api_keys or {}
        
        # Setup additional exchanges for arbitrage
        self.exchanges = {}
        if exchanges:
            for exchange_id, credentials in exchanges.items():
                exchange_class = getattr(ccxt, exchange_id)
                self.exchanges[exchange_id] = exchange_class({
                    'apiKey': credentials['api_key'],
                    'secret': credentials['api_secret'],
                    'enableRateLimit': True
                })
        
        # Add Binance to exchanges dict
        self.exchanges['binance'] = self.client
        
        # Determine symbols to trade
        if symbols:
            self.symbols = symbols
        else:
            # Get top 5 USDT pairs by volume
            exchange_info = self.client.get_exchange_info()
            tickers = self.client.get_ticker()
            usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
            usdt_pairs.sort(key=lambda x: float(x['volume']), reverse=True)
            self.symbols = [p['symbol'] for p in usdt_pairs[:5]]
        
        logger.info(f"Bot initialized with symbols: {self.symbols}")
        
        # ML models
        self.ml_models = {}
        if self.use_ml:
            self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for each symbol and strategy"""
        for symbol in self.symbols:
            self.ml_models[symbol] = {
                'order_book': None,
                'technical': None,
                'combined': None
            }
            
            # Load historical data for training
            try:
                # Get last 1000 1-minute klines for initial training
                klines = self.client.get_historical_klines(
                    symbol, Client.KLINE_INTERVAL_1MINUTE, "1000 minutes ago UTC"
                )
                
                if len(klines) > 100:  # Ensure we have enough data
                    df = self._prepare_training_data(symbol, klines)
                    self._train_models(symbol, df)
                    logger.info(f"ML models trained for {symbol}")
                else:
                    logger.warning(f"Not enough data to train ML models for {symbol}")
            except Exception as e:
                logger.error(f"Error training ML models for {symbol}: {e}")
    
    def _prepare_training_data(self, symbol, klines):
        """Prepare historical data for ML training"""
        # Convert klines to DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert string values to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Calculate features (similar to what we'll use in real-time)
        df['returns'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Technical indicators
        # SMA
        df['sma5'] = df['close'].rolling(5).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma_ratio'] = df['sma5'] / df['sma20']
        
        # Bollinger Bands
        df['std20'] = df['close'].rolling(20).std()
        df['upper_band'] = df['sma20'] + (df['std20'] * 2)
        df['lower_band'] = df['sma20'] - (df['std20'] * 2)
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['sma20']
        df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
        
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Target: Will price increase by target_gain within next 60 mins?
        future_prices = df['close'].shift(-60)  # Look 60 periods ahead (60 mins)
        df['target'] = (future_prices / df['close'] - 1 > self.target_gain).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def _train_models(self, symbol, df):
        """Train ML models using prepared data"""
        # Split data for training
        X = df[[
            'sma_ratio', 'bb_width', 'bb_position', 'rsi', 
            'macd', 'macd_hist', 'volume_change'
        ]]
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Store model and scaler
        self.ml_models[symbol]['technical'] = {
            'model': rf_model,
            'scaler': scaler,
            'features': X.columns.tolist()
        }
        
        # Also train a simpler GradientBoosting model for order book data
        # (features will be constructed in real-time)
        gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        # This is a placeholder - we'll train with actual order book data once collected
        self.ml_models[symbol]['order_book'] = {
            'model': gb_model,
            'features': ['imbalance', 'spread_pct', 'depth_ratio']
        }
        
        logger.info(f"Models trained for {symbol} with accuracy: {rf_model.score(X_test_scaled, y_test):.4f}")

    def start(self):
        """Start the trading bot"""
        if self.running:
            logger.warning("Bot already running")
            return
        
        self.running = True
        
        # Initialize Socket Manager
        self.bsm = BinanceSocketManager(self.client)
        
        # Start data collection threads
        threading.Thread(target=self._collect_order_book_data, daemon=True).start()
        threading.Thread(target=self._collect_technical_data, daemon=True).start()
        
        # If using arbitrage, start exchange data collection
        if len(self.exchanges) > 1:
            threading.Thread(target=self._collect_exchange_data, daemon=True).start()
        
        # If pump and dump detection is enabled, start related threads
        if self.enable_pump_dump_detection:
            threading.Thread(target=self._collect_volume_data, daemon=True).start()
            threading.Thread(target=self._collect_social_sentiment, daemon=True).start()
            threading.Thread(target=self._run_pump_dump_detection, daemon=True).start()
        
        # Start the main trading loop
        threading.Thread(target=self._trading_loop, daemon=True).start()
        
        # Start the trade monitoring loop
        threading.Thread(target=self._monitor_trades, daemon=True).start()
        
        logger.info("Bot started successfully")
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        if self.bsm:
            self.bsm.close()
        logger.info("Bot stopped")
    
    def _collect_order_book_data(self):
        """Collect real-time order book data"""
        connections = {}
        
        for symbol in self.symbols:
            # Initialize order book data structure
            self.order_book_data[symbol] = {
                'bids': {},
                'asks': {},
                'timestamp': 0,
                'computed': {
                    'imbalance': 0,
                    'spread': 0,
                    'spread_pct': 0,
                    'depth_ratio': 0,
                    'top_levels': []
                }
            }
            
            # Start order book web socket
            connections[symbol] = self.bsm.start_depth_socket(
                symbol, self._process_depth_message, depth=BinanceSocketManager.DEPTH_5
            )
        
        # Start socket manager
        self.bsm.start()
        
        logger.info(f"Started collecting order book data for {len(self.symbols)} symbols")
        
        # Keep the thread alive
        while self.running:
            time.sleep(1)
    
    def _process_depth_message(self, msg):
        """Process depth (order book) messages from websocket"""
        if 'e' not in msg or msg['e'] != 'depthUpdate':
            return
        
        symbol = msg['s']
        
        with self.lock:
            order_book = self.order_book_data.get(symbol)
            if not order_book:
                return
            
            # Update bids
            for bid in msg['b']:
                price, quantity = float(bid[0]), float(bid[1])
                if quantity == 0:
                    if price in order_book['bids']:
                        del order_book['bids'][price]
                else:
                    order_book['bids'][price] = quantity
            
            # Update asks
            for ask in msg['a']:
                price, quantity = float(ask[0]), float(ask[1])
                if quantity == 0:
                    if price in order_book['asks']:
                        del order_book['asks'][price]
                else:
                    order_book['asks'][price] = quantity
            
            # Update timestamp
            order_book['timestamp'] = msg['E']
            
            # Compute derived metrics
            self._compute_order_book_metrics(symbol)
    
    def _compute_order_book_metrics(self, symbol):
        """Compute derived metrics from order book data"""
        order_book = self.order_book_data[symbol]
        
        # Get sorted bids and asks
        bids = sorted(order_book['bids'].items(), key=lambda x: x[0], reverse=True)
        asks = sorted(order_book['asks'].items(), key=lambda x: x[0])
        
        if not bids or not asks:
            return
        
        # Get best bid and ask
        best_bid, best_bid_qty = bids[0]
        best_ask, best_ask_qty = asks[0]
        
        # Calculate spread
        spread = best_ask - best_bid
        spread_pct = spread / best_bid
        
        # Calculate imbalance (using top 5 levels or all available if fewer)
        bid_vol = sum(qty for _, qty in bids[:5])
        ask_vol = sum(qty for _, qty in asks[:5])
        
        # Order book imbalance formula from the paper
        if bid_vol + ask_vol > 0:
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        else:
            imbalance = 0
        
        # Calculate depth ratio (volume at best bid vs best ask)
        if best_ask_qty > 0:
            depth_ratio = best_bid_qty / best_ask_qty
        else:
            depth_ratio = 1
        
        # Store top 5 price levels for visualization/analysis
        top_levels = {
            'bids': bids[:5],
            'asks': asks[:5]
        }
        
        # Update computed metrics
        order_book['computed'] = {
            'imbalance': imbalance,
            'spread': spread,
            'spread_pct': spread_pct,
            'depth_ratio': depth_ratio,
            'top_levels': top_levels,
            'last_update': time.time()
        }
    
    def _collect_technical_data(self):
        """Collect and compute technical indicators from klines data"""
        while self.running:
            for symbol in self.symbols:
                try:
                    # Get the most recent 100 1-minute candles
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=Client.KLINE_INTERVAL_1MINUTE,
                        limit=100
                    )
                    
                    # Convert to dataframe
                    df = pd.DataFrame(klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert string values to numeric
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Calculate indicators
                    # SMA
                    df['sma5'] = df['close'].rolling(5).mean()
                    df['sma20'] = df['close'].rolling(20).mean()
                    df['sma_ratio'] = df['sma5'] / df['sma20']
                    
                    # Bollinger Bands
                    df['sma20'] = df['close'].rolling(20).mean()
                    df['std20'] = df['close'].rolling(20).std()
                    df['upper_band'] = df['sma20'] + (df['std20'] * 2)
                    df['lower_band'] = df['sma20'] - (df['std20'] * 2)
                    df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['sma20']
                    df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
                    
                    # RSI (14)
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    
                    # MACD
                    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
                    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
                    df['macd'] = df['ema12'] - df['ema26']
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
                    
                    # Volume analysis
                    df['volume_sma20'] = df['volume'].rolling(20).mean()
                    df['volume_ratio'] = df['volume'] / df['volume_sma20']
                    
                    # Price momentum
                    df['price_change_1m'] = df['close'].pct_change(1)
                    df['price_change_5m'] = df['close'].pct_change(5)
                    
                    # Store most recent data
                    with self.lock:
                        # Extract the most recent row with complete indicators
                        latest_data = df.iloc[-1].to_dict()
                        self.technical_data[symbol] = {
                            'data': latest_data,
                            'timestamp': time.time(),
                            'dataframe': df.tail(20)  # Keep recent history for analysis
                        }
                
                except Exception as e:
                    logger.error(f"Error collecting technical data for {symbol}: {e}")
            
            # Update every 30 seconds
            time.sleep(30)
    
    def _collect_exchange_data(self):
        """Collect price data from multiple exchanges for arbitrage"""
        while self.running:
            for symbol in self.symbols:
                symbol_data = {}
                
                # For each exchange, get ticker data
                for exchange_id, exchange in self.exchanges.items():
                    try:
                        if exchange_id == 'binance':
                            # For Binance, we use the Client
                            ticker = self.client.get_symbol_ticker(symbol=symbol)
                            price = float(ticker['price'])
                        else:
                            # For other exchanges, we use ccxt
                            # May need to standardize the symbol format for different exchanges
                            standardized_symbol = symbol.replace('USDT', '/USDT')
                            ticker = exchange.fetch_ticker(standardized_symbol)
                            price = ticker['last']
                        
                        symbol_data[exchange_id] = {
                            'price': price,
                            'timestamp': time.time()
                        }
                        
                    except Exception as e:
                        logger.error(f"Error fetching {symbol} price from {exchange_id}: {e}")
                
                # Store the exchange data
                with self.lock:
                    self.exchange_price_data[symbol] = symbol_data
            
            # Check for arbitrage opportunities every 5 seconds
            time.sleep(5)
    
    def _trading_loop(self):
        """Main trading loop that evaluates strategies and executes trades"""
        while self.running:
            # Skip if we're at max open trades
            if len(self.active_trades) >= self.max_open_trades:
                time.sleep(5)
                continue
            
            # Analyze each symbol
            for symbol in self.symbols:
                # Skip if we already have an active trade for this symbol
                if symbol in self.active_trades:
                    continue
                
                try:
                    # Get signals from different strategies
                    order_book_signal = self._get_order_book_signal(symbol)
                    technical_signal = self._get_technical_signal(symbol)
                    arbitrage_signal = self._get_arbitrage_signal(symbol)
                    
                    # Get pump and dump signal if enabled
                    pump_dump_signal = self._get_pump_dump_signal(symbol) if self.enable_pump_dump_detection else {'signal': 'neutral', 'confidence': 0, 'data': {}}
                    
                    # Combine signals and make decision
                    decision, confidence, strategy = self._make_trading_decision(
                        symbol, order_book_signal, technical_signal, arbitrage_signal, pump_dump_signal
                    )
                    
                    # Execute trade if decision is positive
                    if decision == 'buy' and confidence > 0.65:  # Confidence threshold
                        self._execute_trade(symbol, 'buy', confidence, strategy)
                
                except Exception as e:
                    logger.error(f"Error in trading loop for {symbol}: {e}")
            
            # Check every 5 seconds
            time.sleep(5)
    
    def _get_order_book_signal(self, symbol):
        """Generate trading signal based on order book imbalance"""
        with self.lock:
            if symbol not in self.order_book_data:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            ob_data = self.order_book_data[symbol]
            
            # Check if data is fresh (less than 5 seconds old)
            if 'computed' not in ob_data or time.time() - ob_data['computed'].get('last_update', 0) > 5:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            computed = ob_data['computed']
            imbalance = computed['imbalance']
            
            # Prepare feature vector for ML if we're using it
            if self.use_ml and symbol in self.ml_models and self.ml_models[symbol]['order_book']:
                features = [
                    imbalance,
                    computed['spread_pct'],
                    computed['depth_ratio']
                ]
                # We would need to have trained this model with actual data
                # This is just a placeholder for now
                # model = self.ml_models[symbol]['order_book']['model']
                # confidence = model.predict_proba([features])[0][1]  # Probability of class 1
                
                # For now, use a rule-based approach
                if imbalance > 0.3:  # Strong buy imbalance
                    signal = 'buy'
                    confidence = min(0.5 + imbalance/2, 0.9)  # Map imbalance to confidence
                elif imbalance < -0.3:  # Strong sell imbalance
                    signal = 'sell'
                    confidence = min(0.5 + abs(imbalance)/2, 0.9)
                else:
                    signal = 'neutral'
                    confidence = 0.5
            else:
                # Rule-based approach without ML
                if imbalance > 0.3:  # Strong buy imbalance
                    signal = 'buy'
                    confidence = min(0.5 + imbalance/2, 0.9)
                elif imbalance < -0.3:  # Strong sell imbalance
                    signal = 'sell'
                    confidence = min(0.5 + abs(imbalance)/2, 0.9)
                else:
                    signal = 'neutral'
                    confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': confidence,
                'data': {
                    'imbalance': imbalance,
                    'spread_pct': computed['spread_pct'],
                    'depth_ratio': computed['depth_ratio']
                }
            }
    
    def _get_technical_signal(self, symbol):
        """Generate trading signal based on technical indicators"""
        with self.lock:
            if symbol not in self.technical_data:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            tech_data = self.technical_data[symbol]
            data = tech_data['data']
            
            # Check if data is fresh (less than 2 minutes old)
            if time.time() - tech_data.get('timestamp', 0) > 120:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            # Use ML model if available
            if (self.use_ml and symbol in self.ml_models and 
                self.ml_models[symbol]['technical'] and 
                'model' in self.ml_models[symbol]['technical']):
                
                model_info = self.ml_models[symbol]['technical']
                model = model_info['model']
                scaler = model_info['scaler']
                features = model_info['features']
                
                # Extract feature values
                feature_values = []
                for feature in features:
                    feature_values.append(data.get(feature, 0))
                
                # Scale features
                scaled_features = scaler.transform([feature_values])
                
                # Get prediction and probability
                prediction = model.predict(scaled_features)[0]
                probabilities = model.predict_proba(scaled_features)[0]
                
                if prediction == 1:  # Buy signal
                    signal = 'buy'
                    confidence = probabilities[1]
                else:
                    signal = 'sell' if probabilities[0] > 0.7 else 'neutral'
                    confidence = probabilities[0] if signal == 'sell' else 0.5
            
            else:
                # Rule-based approach
                # Check for bullish conditions
                bullish_conditions = [
                    data.get('sma_ratio', 1) > 1.005,  # Fast SMA above slow SMA
                    data.get('macd', 0) > data.get('macd_signal', 0),  # MACD above signal
                    data.get('rsi', 50) > 50 and data.get('rsi', 50) < 70,  # RSI in bullish zone but not overbought
                    data.get('bb_position', 0.5) > 0.5 and data.get('bb_position', 0.5) < 0.9,  # Price in upper half of BB but not at extreme
                    data.get('volume_ratio', 1) > 1.2  # Higher than average volume
                ]
                
                # Check for bearish conditions
                bearish_conditions = [
                    data.get('sma_ratio', 1) < 0.995,  # Fast SMA below slow SMA
                    data.get('macd', 0) < data.get('macd_signal', 0),  # MACD below signal
                    data.get('rsi', 50) < 50 and data.get('rsi', 50) > 30,  # RSI in bearish zone but not oversold
                    data.get('bb_position', 0.5) < 0.5 and data.get('bb_position', 0.5) > 0.1,  # Price in lower half of BB but not at extreme
                    data.get('volume_ratio', 1) > 1.2  # Higher than average volume
                ]
                
                bullish_count = sum(bullish_conditions)
                bearish_count = sum(bearish_conditions)
                
                # Determine signal and confidence
                if bullish_count >= 3 and bullish_count > bearish_count + 1:
                    signal = 'buy'
                    confidence = 0.5 + (bullish_count / 10)
                elif bearish_count >= 3 and bearish_count > bullish_count + 1:
                    signal = 'sell'
                    confidence = 0.5 + (bearish_count / 10)
                else:
                    signal = 'neutral'
                    confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': confidence,
                'data': {
                    'sma_ratio': data.get('sma_ratio', 1),
                    'macd': data.get('macd', 0),
                    'rsi': data.get('rsi', 50),
                    'bb_position': data.get('bb_position', 0.5),
                    'volume_ratio': data.get('volume_ratio', 1)
                }
            }
    
    def _get_arbitrage_signal(self, symbol):
        """Generate trading signal based on arbitrage opportunities"""
        with self.lock:
            if symbol not in self.exchange_price_data or len(self.exchange_price_data[symbol]) < 2:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            symbol_data = self.exchange_price_data[symbol]
            
            # Check all prices are recent (less than 30 seconds old)
            for exchange, data in symbol_data.items():
                if time.time() - data.get('timestamp', 0) > 30:
                    return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            # Find min and max prices across exchanges
            prices = [(exchange, data['price']) for exchange, data in symbol_data.items()]
            min_exchange, min_price = min(prices, key=lambda x: x[1])
            max_exchange, max_price = max(prices, key=lambda x: x[1])
            
            # Calculate price difference percentage
            price_diff_pct = (max_price - min_price) / min_price
            
            # Factor in trading fees
            # Assume 0.1% fee per trade (0.2% round trip)
            arbitrage_threshold = 0.002  # 0.2% minimum profitable arbitrage
            
            # Determine signal and confidence
            if price_diff_pct > arbitrage_threshold:
                # Potential arbitrage opportunity
                signal = 'buy'  # Buy on cheaper exchange
                # Scale confidence based on how much the spread exceeds our threshold
                confidence = min(0.5 + (price_diff_pct - arbitrage_threshold) * 50, 0.95)
                
                # Store which exchange to buy on and which to sell on
                buy_exchange = min_exchange
                sell_exchange = max_exchange
            else:
                signal = 'neutral'
                confidence = 0.5
                buy_exchange = None
                sell_exchange = None
            
            return {
                'signal': signal,
                'confidence': confidence,
                'data': {
                    'price_diff_pct': price_diff_pct,
                    'buy_exchange': buy_exchange,
                    'sell_exchange': sell_exchange,
                    'min_price': min_price,
                    'max_price': max_price
                }
            }, '', symbol)
                if not base_symbol:
                    continue
                
                symbol_sentiment = {
                    'combined_score': 0,
                    'sentiment_count': 0,
                    'source_data': {}
                }
                
                # Combine sentiment from all sources
                for source, sentiments in all_sentiments.items():
                    if base_symbol in sentiments:
                        source_sentiment = sentiments[base_symbol]
                        symbol_sentiment['source_data'][source] = source_sentiment
                        
                        # Add to combined score
                        if 'compound' in source_sentiment and source_sentiment['count'] > 0:
                            symbol_sentiment['combined_score'] += source_sentiment['compound'] * source_sentiment['count']
                            symbol_sentiment['sentiment_count'] += source_sentiment['count']
                
                # Calculate average compound score
                if symbol_sentiment['sentiment_count'] > 0:
                    symbol_sentiment['avg_compound'] = symbol_sentiment['combined_score'] / symbol_sentiment['sentiment_count']
                else:
                    symbol_sentiment['avg_compound'] = 0
                
                # Store sentiment data
                with self.lock:
                    self.social_sentiment_data[symbol] = {
                        'sentiment': symbol_sentiment,
                        'timestamp': time.time()
                    }
            
            # Check every 5 minutes
            time.sleep(300)
    
    def _get_twitter_sentiment(self, symbols, sentiment_analyzer):
        """Get Twitter sentiment for a list of cryptocurrency symbols"""
        result = {}
        
        # Set up Twitter API client
        api_keys = self.social_api_keys['twitter']
        try:
            auth = tweepy.OAuthHandler(api_keys['consumer_key'], api_keys['consumer_secret'])
            auth.set_access_token(api_keys['access_token'], api_keys['access_token_secret'])
            api = tweepy.API(auth, wait_on_rate_limit=True)
        except:
            logger.error("Failed to initialize Twitter API client")
            return result
        
        for symbol in symbols:
            try:
                # Search for tweets containing the symbol
                search_terms = [f"#{symbol}", f"${symbol}", symbol]
                all_tweets = []
                
                for term in search_terms:
                    tweets = api.search_tweets(q=term, count=100, lang="en", tweet_mode="extended")
                    all_tweets.extend([tweet.full_text for tweet in tweets])
                
                # Remove duplicates
                all_tweets = list(set(all_tweets))
                
                if not all_tweets:
                    continue
                
                # Calculate sentiment scores
                compound_scores = []
                for tweet in all_tweets:
                    sentiment = sentiment_analyzer.polarity_scores(tweet)
                    compound_scores.append(sentiment['compound'])
                
                # Calculate average sentiment
                avg_compound = np.mean(compound_scores) if compound_scores else 0
                
                # Store results
                result[symbol] = {
                    'compound': avg_compound,
                    'count': len(all_tweets),
                    'positive_pct': sum(1 for score in compound_scores if score > 0.05) / len(compound_scores) if compound_scores else 0,
                    'negative_pct': sum(1 for score in compound_scores if score < -0.05) / len(compound_scores) if compound_scores else 0,
                    'neutral_pct': sum(1 for score in compound_scores if -0.05 <= score <= 0.05) / len(compound_scores) if compound_scores else 0
                }
            except Exception as e:
                logger.error(f"Error getting Twitter sentiment for {symbol}: {e}")
        
        return result
    
    def _get_reddit_sentiment(self, symbols, sentiment_analyzer):
        """Get Reddit sentiment for a list of cryptocurrency symbols"""
        result = {}
        
        # Set up Reddit API client
        api_keys = self.social_api_keys['reddit']
        try:
            reddit = praw.Reddit(
                client_id=api_keys['client_id'],
                client_secret=api_keys['client_secret'],
                user_agent=api_keys['user_agent']
            )
        except:
            logger.error("Failed to initialize Reddit API client")
            return result
        
        # Subreddits to check
        subreddits = ['CryptoCurrency', 'CryptoMarkets', 'SatoshiStreetBets']
        
        for symbol in symbols:
            try:
                all_posts = []
                all_comments = []
                
                for subreddit_name in subreddits:
                    subreddit = reddit.subreddit(subreddit_name)
                    
                    # Get posts
                    for post in subreddit.search(symbol, limit=50, time_filter='day'):
                        all_posts.append(post.title)
                        if post.selftext:
                            all_posts.append(post.selftext)
                        
                        # Get comments from post
                        post.comments.replace_more(limit=0)
                        for comment in post.comments.list():
                            all_comments.append(comment.body)
                
                # Combine posts and comments
                all_content = all_posts + all_comments
                
                if not all_content:
                    continue
                
                # Calculate sentiment scores
                compound_scores = []
                for content in all_content:
                    sentiment = sentiment_analyzer.polarity_scores(content)
                    compound_scores.append(sentiment['compound'])
                
                # Calculate average sentiment
                avg_compound = np.mean(compound_scores) if compound_scores else 0
                
                # Store results
                result[symbol] = {
                    'compound': avg_compound,
                    'count': len(all_content),
                    'positive_pct': sum(1 for score in compound_scores if score > 0.05) / len(compound_scores) if compound_scores else 0,
                    'negative_pct': sum(1 for score in compound_scores if score < -0.05) / len(compound_scores) if compound_scores else 0,
                    'neutral_pct': sum(1 for score in compound_scores if -0.05 <= score <= 0.05) / len(compound_scores) if compound_scores else 0
                }
            except Exception as e:
                logger.error(f"Error getting Reddit sentiment for {symbol}: {e}")
        
        return result
        
    def _run_pump_dump_detection(self):
        """Run pump and dump detection algorithm periodically"""
        # Wait for initial data collection
        time.sleep(60)
        
        # Initialize the pump and dump detector model
        if self.use_ml:
            self._initialize_pump_dump_model()
        
        while self.running:
            for symbol in self.symbols:
                try:
                    # Analyze the symbol for pump and dump patterns
                    result = self._analyze_pump_dump_patterns(symbol)
                    
                    # Log significant detections
                    if result and result['signal'] != 'neutral' and result['confidence'] > 0.75:
                        if result['signal'] == 'buy':
                            logger.info(f"Detected potential pump in {symbol} with confidence {result['confidence']:.2f}")
                        elif result['signal'] == 'sell':
                            logger.info(f"Detected potential dump in {symbol} with confidence {result['confidence']:.2f}")
                
                except Exception as e:
                    logger.error(f"Error in pump and dump detection for {symbol}: {e}")
            
            # Run every minute
            time.sleep(60)
    
    def _initialize_pump_dump_model(self):
        """Initialize the machine learning model for pump and dump detection"""
        try:
            # Initialize a Random Forest Classifier for pump detection
            self.pump_detect_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )
            
            # We'll train the model if historical data is available
            # For now, this is a placeholder as we'll use rule-based approach initially
            logger.info("Initialized pump and dump detection model")
        except Exception as e:
            logger.error(f"Error initializing pump and dump model: {e}")
            self.pump_detect_model = None
    
    def _analyze_pump_dump_patterns(self, symbol):
        """Analyze a symbol for pump and dump patterns"""
        with self.lock:
            # Check if we have the required data
            if (symbol not in self.pump_dump_data or 
                symbol not in self.technical_data or 
                symbol not in self.order_book_data):
                return None
            
            # Get data
            volume_data = self.pump_dump_data[symbol].get('volume_data', {})
            tech_data = self.technical_data[symbol].get('data', {})
            order_book = self.order_book_data[symbol]
            sentiment_data = self.social_sentiment_data.get(symbol, {}).get('sentiment', {})
            
            # Rule-based approach for pump detection
            pump_signals = []
            dump_signals = []
            
            # Check volume anomalies across different timeframes
            for tf, data in volume_data.items():
                # Volume ratio (current volume compared to average)
                if data['volume_ratio'] > 3:  # Volume 3x higher than average
                    pump_signals.append({
                        'indicator': f'volume_spike_{tf}',
                        'value': data['volume_ratio'],
                        'weight': 0.7
                    })
                
                # Volume acceleration (increasing rate of volume)
                if data['volume_accel'] > 0.5:  # Significant acceleration in volume
                    pump_signals.append({
                        'indicator': f'volume_accel_{tf}',
                        'value': data['volume_accel'],
                        'weight': 0.6
                    })
            
            # Check price momentum from technical data
            if 'price_change_1m' in tech_data and tech_data['price_change_1m'] > 0.02:  # 2% price rise in 1 min
                pump_signals.append({
                    'indicator': 'price_spike_1m',
                    'value': tech_data['price_change_1m'],
                    'weight': 0.8
                })
            
            if 'price_change_5m' in tech_data and tech_data['price_change_5m'] > 0.05:  # 5% price rise in 5 min
                pump_signals.append({
                    'indicator': 'price_spike_5m',
                    'value': tech_data['price_change_5m'],
                    'weight': 0.75
                })
            
            # Check for RSI overbought conditions (potential dump signal)
            if 'rsi' in tech_data and tech_data['rsi'] > 80:  # Overbought RSI
                dump_signals.append({
                    'indicator': 'overbought_rsi',
                    'value': tech_data['rsi'],
                    'weight': 0.65
                })
            
            # Check order book imbalance
            if 'computed' in order_book and 'imbalance' in order_book['computed']:
                imbalance = order_book['computed']['imbalance']
                if imbalance > 0.5:  # Strong buy imbalance, potential pump
                    pump_signals.append({
                        'indicator': 'order_book_imbalance',
                        'value': imbalance,
                        'weight': 0.6
                    })
                elif imbalance < -0.5:  # Strong sell imbalance, potential dump
                    dump_signals.append({
                        'indicator': 'order_book_imbalance',
                        'value': imbalance,
                        'weight': 0.6
                    })
            
            # Check social sentiment if available
            if sentiment_data:
                # High positive sentiment and high message count might indicate a pump
                if (sentiment_data.get('avg_compound', 0) > 0.5 and 
                    sentiment_data.get('sentiment_count', 0) > 50):
                    pump_signals.append({
                        'indicator': 'positive_sentiment_spike',
                        'value': sentiment_data.get('avg_compound', 0),
                        'weight': 0.55
                    })
            
            # Calculate combined signal strengths
            pump_strength = sum(signal['weight'] * min(1, abs(signal['value'])) 
                              for signal in pump_signals) / max(1, len(pump_signals))
            
            dump_strength = sum(signal['weight'] * min(1, abs(signal['value'])) 
                              for signal in dump_signals) / max(1, len(dump_signals))
            
            # Determine final signal
            if pump_signals and pump_strength > 0.6:
                signal = 'buy'
                confidence = min(0.95, pump_strength)
                signal_type = 'pump_detected'
            elif dump_signals and dump_strength > 0.6:
                signal = 'sell'
                confidence = min(0.95, dump_strength)
                signal_type = 'dump_detected'
            else:
                signal = 'neutral'
                confidence = 0.5
                signal_type = 'normal'
            
            # Return the result
            return {
                'signal': signal,
                'confidence': confidence,
                'type': signal_type,
                'pump_signals': pump_signals,
                'dump_signals': dump_signals,
                'timestamp': time.time()
            }
    
    def _get_pump_dump_signal(self, symbol):
        """Generate trading signal based on pump and dump detection"""
        try:
            # Run the pump and dump analysis
            result = self._analyze_pump_dump_patterns(symbol)
            
            if not result:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            # Return the signal
            return {
                'signal': result['signal'],
                'confidence': result['confidence'],
                'data': {
                    'type': result['type'],
                    'pump_signals': [s['indicator'] for s in result.get('pump_signals', [])],
                    'dump_signals': [s['indicator'] for s in result.get('dump_signals', [])]
                }
            }
        except Exception as e:
            logger.error(f"Error getting pump dump signal for {symbol}: {e}")
            return {'signal': 'neutral', 'confidence': 0, 'data': {}}
    
    def _adjust_trade_for_pump_dump(self, symbol, trade_params):
        """Adjust trade parameters based on pump and dump detection"""
        try:
            # Get pump dump signal
            pd_signal = self._get_pump_dump_signal(symbol)
            
            if pd_signal['signal'] == 'neutral' or pd_signal['confidence'] < 0.7:
                return trade_params
            
            # If strong pump signal, adjust trade parameters
            if pd_signal['signal'] == 'buy':
                # Set more aggressive take profit (higher %)
                trade_params['target_gain'] = self.target_gain * 1.5
                
                # Set tighter stop loss 
                trade_params['stop_loss_pct'] = 0.02  # 2% stop loss
                
                # Set shorter maximum duration
                trade_params['max_duration_minutes'] = min(30, self.timeframe_minutes)
                
                logger.info(f"Adjusted trade parameters for pump in {symbol}")
            
            # If strong dump signal, either avoid trade or set different parameters
            elif pd_signal['signal'] == 'sell':
                # Option 1: Avoid trade entirely
                trade_params['skip_trade'] = True
                
                logger.info(f"Avoiding trade due to dump signal in {symbol}")
            
            return trade_params
        except Exception as e:
            logger.error(f"Error adjusting trade for pump/dump in {symbol}: {e}")
            return trade_params"""
Advanced Binance Trading Bot for Ultra-Short-Term Trading (30-60 min windows)
This bot implements multiple high-frequency trading strategies:
1. Order Book Imbalance Detection
2. Technical Indicator Momentum Scalping
3. Cross-Exchange Arbitrage
4. Statistical Arbitrage (Mean Reversion)
5. Pump and Dump Detection and Trading

Features:
- Real-time order book analysis
- ML-enhanced signal validation
- Adaptive position sizing
- Risk management with tight stop-losses
- Multi-strategy execution
- Social media sentiment analysis
- Volume anomaly detection
- Pattern recognition for pump and dump schemes

Requirements:
pip install python-binance pandas numpy scikit-learn tensorflow websocket-client ccxt nltk vaderSentiment requests tweepy praw
"""

import os
import time
import numpy as np
import pandas as pd
import threading
import json
import logging
import datetime
import warnings
import math
import csv
import re
import requests
import tweepy
import praw
from collections import defaultdict, Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentimentAnalyzer
from binance.client import Client
from binance.websockets import BinanceSocketManager
from binance.enums import *
import ccxt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

class BinanceHFTBot:
    def __init__(self, api_key, api_secret, target_gain=0.03, timeframe_minutes=60, 
                 base_trade_amount=100, max_open_trades=3, use_ml=True, 
                 exchanges=None, symbols=None, 
                 social_api_keys=None, enable_pump_dump_detection=True):
        """
        Initialize the high-frequency trading bot
        
        Parameters:
        - api_key: Binance API key
        - api_secret: Binance API secret
        - target_gain: Target gain percentage (default 3%)
        - timeframe_minutes: Maximum trade duration (default 60 minutes)
        - base_trade_amount: Base amount to trade in USDT
        - max_open_trades: Maximum number of concurrent trades
        - use_ml: Whether to use machine learning for signal validation
        - exchanges: List of additional exchanges for arbitrage (ccxt format)
        - symbols: List of trading pairs to monitor (default: top 5 by volume)
        - social_api_keys: API keys for social media platforms
        - enable_pump_dump_detection: Whether to enable pump and dump detection
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = Client(api_key, api_secret)
        self.bsm = None
        self.start_time = time.time()  # Record when the bot starts
        
        # Bot settings
        self.target_gain = target_gain
        self.timeframe_minutes = timeframe_minutes
        self.base_trade_amount = base_trade_amount
        self.max_open_trades = max_open_trades
        self.use_ml = use_ml
        self.enable_pump_dump_detection = enable_pump_dump_detection
        
        # State variables
        self.active_trades = {}
        self.order_book_data = {}
        self.technical_data = {}
        self.exchange_price_data = {}
        self.pump_dump_data = {}
        self.social_sentiment_data = {}
        self.running = False
        self.lock = threading.Lock()
        
        # Social API keys
        self.social_api_keys = social_api_keys or {}
        
        # Setup additional exchanges for arbitrage
        self.exchanges = {}
        if exchanges:
            for exchange_id, credentials in exchanges.items():
                exchange_class = getattr(ccxt, exchange_id)
                self.exchanges[exchange_id] = exchange_class({
                    'apiKey': credentials['api_key'],
                    'secret': credentials['api_secret'],
                    'enableRateLimit': True
                })
        
        # Add Binance to exchanges dict
        self.exchanges['binance'] = self.client
        
        # Determine symbols to trade
        if symbols:
            self.symbols = symbols
        else:
            # Get top 5 USDT pairs by volume
            exchange_info = self.client.get_exchange_info()
            tickers = self.client.get_ticker()
            usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
            usdt_pairs.sort(key=lambda x: float(x['volume']), reverse=True)
            self.symbols = [p['symbol'] for p in usdt_pairs[:5]]
        
        logger.info(f"Bot initialized with symbols: {self.symbols}")
        
        # ML models
        self.ml_models = {}
        if self.use_ml:
            self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for each symbol and strategy"""
        for symbol in self.symbols:
            self.ml_models[symbol] = {
                'order_book': None,
                'technical': None,
                'combined': None
            }
            
            # Load historical data for training
            try:
                # Get last 1000 1-minute klines for initial training
                klines = self.client.get_historical_klines(
                    symbol, Client.KLINE_INTERVAL_1MINUTE, "1000 minutes ago UTC"
                )
                
                if len(klines) > 100:  # Ensure we have enough data
                    df = self._prepare_training_data(symbol, klines)
                    self._train_models(symbol, df)
                    logger.info(f"ML models trained for {symbol}")
                else:
                    logger.warning(f"Not enough data to train ML models for {symbol}")
            except Exception as e:
                logger.error(f"Error training ML models for {symbol}: {e}")
    
    def _prepare_training_data(self, symbol, klines):
        """Prepare historical data for ML training"""
        # Convert klines to DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert string values to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Calculate features (similar to what we'll use in real-time)
        df['returns'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Technical indicators
        # SMA
        df['sma5'] = df['close'].rolling(5).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma_ratio'] = df['sma5'] / df['sma20']
        
        # Bollinger Bands
        df['std20'] = df['close'].rolling(20).std()
        df['upper_band'] = df['sma20'] + (df['std20'] * 2)
        df['lower_band'] = df['sma20'] - (df['std20'] * 2)
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['sma20']
        df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
        
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Target: Will price increase by target_gain within next 60 mins?
        future_prices = df['close'].shift(-60)  # Look 60 periods ahead (60 mins)
        df['target'] = (future_prices / df['close'] - 1 > self.target_gain).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def _train_models(self, symbol, df):
        """Train ML models using prepared data"""
        # Split data for training
        X = df[[
            'sma_ratio', 'bb_width', 'bb_position', 'rsi', 
            'macd', 'macd_hist', 'volume_change'
        ]]
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Store model and scaler
        self.ml_models[symbol]['technical'] = {
            'model': rf_model,
            'scaler': scaler,
            'features': X.columns.tolist()
        }
        
        # Also train a simpler GradientBoosting model for order book data
        # (features will be constructed in real-time)
        gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        # This is a placeholder - we'll train with actual order book data once collected
        self.ml_models[symbol]['order_book'] = {
            'model': gb_model,
            'features': ['imbalance', 'spread_pct', 'depth_ratio']
        }
        
        logger.info(f"Models trained for {symbol} with accuracy: {rf_model.score(X_test_scaled, y_test):.4f}")

    def start(self):
        """Start the trading bot"""
        if self.running:
            logger.warning("Bot already running")
            return
        
        self.running = True
        
        # Initialize Socket Manager
        self.bsm = BinanceSocketManager(self.client)
        
        # Start data collection threads
        threading.Thread(target=self._collect_order_book_data, daemon=True).start()
        threading.Thread(target=self._collect_technical_data, daemon=True).start()
        
        # If using arbitrage, start exchange data collection
        if len(self.exchanges) > 1:
            threading.Thread(target=self._collect_exchange_data, daemon=True).start()
        
        # If pump and dump detection is enabled, start related threads
        if self.enable_pump_dump_detection:
            threading.Thread(target=self._collect_volume_data, daemon=True).start()
            threading.Thread(target=self._collect_social_sentiment, daemon=True).start()
            threading.Thread(target=self._run_pump_dump_detection, daemon=True).start()
        
        # Start the main trading loop
        threading.Thread(target=self._trading_loop, daemon=True).start()
        
        # Start the trade monitoring loop
        threading.Thread(target=self._monitor_trades, daemon=True).start()
        
        logger.info("Bot started successfully")
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        if self.bsm:
            self.bsm.close()
        logger.info("Bot stopped")
    
    def _collect_order_book_data(self):
        """Collect real-time order book data"""
        connections = {}
        
        for symbol in self.symbols:
            # Initialize order book data structure
            self.order_book_data[symbol] = {
                'bids': {},
                'asks': {},
                'timestamp': 0,
                'computed': {
                    'imbalance': 0,
                    'spread': 0,
                    'spread_pct': 0,
                    'depth_ratio': 0,
                    'top_levels': []
                }
            }
            
            # Start order book web socket
            connections[symbol] = self.bsm.start_depth_socket(
                symbol, self._process_depth_message, depth=BinanceSocketManager.DEPTH_5
            )
        
        # Start socket manager
        self.bsm.start()
        
        logger.info(f"Started collecting order book data for {len(self.symbols)} symbols")
        
        # Keep the thread alive
        while self.running:
            time.sleep(1)
    
    def _process_depth_message(self, msg):
        """Process depth (order book) messages from websocket"""
        if 'e' not in msg or msg['e'] != 'depthUpdate':
            return
        
        symbol = msg['s']
        
        with self.lock:
            order_book = self.order_book_data.get(symbol)
            if not order_book:
                return
            
            # Update bids
            for bid in msg['b']:
                price, quantity = float(bid[0]), float(bid[1])
                if quantity == 0:
                    if price in order_book['bids']:
                        del order_book['bids'][price]
                else:
                    order_book['bids'][price] = quantity
            
            # Update asks
            for ask in msg['a']:
                price, quantity = float(ask[0]), float(ask[1])
                if quantity == 0:
                    if price in order_book['asks']:
                        del order_book['asks'][price]
                else:
                    order_book['asks'][price] = quantity
            
            # Update timestamp
            order_book['timestamp'] = msg['E']
            
            # Compute derived metrics
            self._compute_order_book_metrics(symbol)
    
    def _compute_order_book_metrics(self, symbol):
        """Compute derived metrics from order book data"""
        order_book = self.order_book_data[symbol]
        
        # Get sorted bids and asks
        bids = sorted(order_book['bids'].items(), key=lambda x: x[0], reverse=True)
        asks = sorted(order_book['asks'].items(), key=lambda x: x[0])
        
        if not bids or not asks:
            return
        
        # Get best bid and ask
        best_bid, best_bid_qty = bids[0]
        best_ask, best_ask_qty = asks[0]
        
        # Calculate spread
        spread = best_ask - best_bid
        spread_pct = spread / best_bid
        
        # Calculate imbalance (using top 5 levels or all available if fewer)
        bid_vol = sum(qty for _, qty in bids[:5])
        ask_vol = sum(qty for _, qty in asks[:5])
        
        # Order book imbalance formula from the paper
        if bid_vol + ask_vol > 0:
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        else:
            imbalance = 0
        
        # Calculate depth ratio (volume at best bid vs best ask)
        if best_ask_qty > 0:
            depth_ratio = best_bid_qty / best_ask_qty
        else:
            depth_ratio = 1
        
        # Store top 5 price levels for visualization/analysis
        top_levels = {
            'bids': bids[:5],
            'asks': asks[:5]
        }
        
        # Update computed metrics
        order_book['computed'] = {
            'imbalance': imbalance,
            'spread': spread,
            'spread_pct': spread_pct,
            'depth_ratio': depth_ratio,
            'top_levels': top_levels,
            'last_update': time.time()
        }
    
    def _collect_technical_data(self):
        """Collect and compute technical indicators from klines data"""
        while self.running:
            for symbol in self.symbols:
                try:
                    # Get the most recent 100 1-minute candles
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=Client.KLINE_INTERVAL_1MINUTE,
                        limit=100
                    )
                    
                    # Convert to dataframe
                    df = pd.DataFrame(klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert string values to numeric
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Calculate indicators
                    # SMA
                    df['sma5'] = df['close'].rolling(5).mean()
                    df['sma20'] = df['close'].rolling(20).mean()
                    df['sma_ratio'] = df['sma5'] / df['sma20']
                    
                    # Bollinger Bands
                    df['sma20'] = df['close'].rolling(20).mean()
                    df['std20'] = df['close'].rolling(20).std()
                    df['upper_band'] = df['sma20'] + (df['std20'] * 2)
                    df['lower_band'] = df['sma20'] - (df['std20'] * 2)
                    df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['sma20']
                    df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
                    
                    # RSI (14)
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    
                    # MACD
                    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
                    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
                    df['macd'] = df['ema12'] - df['ema26']
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
                    
                    # Volume analysis
                    df['volume_sma20'] = df['volume'].rolling(20).mean()
                    df['volume_ratio'] = df['volume'] / df['volume_sma20']
                    
                    # Price momentum
                    df['price_change_1m'] = df['close'].pct_change(1)
                    df['price_change_5m'] = df['close'].pct_change(5)
                    
                    # Store most recent data
                    with self.lock:
                        # Extract the most recent row with complete indicators
                        latest_data = df.iloc[-1].to_dict()
                        self.technical_data[symbol] = {
                            'data': latest_data,
                            'timestamp': time.time(),
                            'dataframe': df.tail(20)  # Keep recent history for analysis
                        }
                
                except Exception as e:
                    logger.error(f"Error collecting technical data for {symbol}: {e}")
            
            # Update every 30 seconds
            time.sleep(30)
    
    def _collect_exchange_data(self):
        """Collect price data from multiple exchanges for arbitrage"""
        while self.running:
            for symbol in self.symbols:
                symbol_data = {}
                
                # For each exchange, get ticker data
                for exchange_id, exchange in self.exchanges.items():
                    try:
                        if exchange_id == 'binance':
                            # For Binance, we use the Client
                            ticker = self.client.get_symbol_ticker(symbol=symbol)
                            price = float(ticker['price'])
                        else:
                            # For other exchanges, we use ccxt
                            # May need to standardize the symbol format for different exchanges
                            standardized_symbol = symbol.replace('USDT', '/USDT')
                            ticker = exchange.fetch_ticker(standardized_symbol)
                            price = ticker['last']
                        
                        symbol_data[exchange_id] = {
                            'price': price,
                            'timestamp': time.time()
                        }
                        
                    except Exception as e:
                        logger.error(f"Error fetching {symbol} price from {exchange_id}: {e}")
                
                # Store the exchange data
                with self.lock:
                    self.exchange_price_data[symbol] = symbol_data
            
            # Check for arbitrage opportunities every 5 seconds
            time.sleep(5)
    
    def _trading_loop(self):
        """Main trading loop that evaluates strategies and executes trades"""
        while self.running:
            # Skip if we're at max open trades
            if len(self.active_trades) >= self.max_open_trades:
                time.sleep(5)
                continue
            
            # Analyze each symbol
            for symbol in self.symbols:
                # Skip if we already have an active trade for this symbol
                if symbol in self.active_trades:
                    continue
                
                try:
                    # Get signals from different strategies
                    order_book_signal = self._get_order_book_signal(symbol)
                    technical_signal = self._get_technical_signal(symbol)
                    arbitrage_signal = self._get_arbitrage_signal(symbol)
                    
                    # Get pump and dump signal if enabled
                    pump_dump_signal = self._get_pump_dump_signal(symbol) if self.enable_pump_dump_detection else {'signal': 'neutral', 'confidence': 0, 'data': {}}
                    
                    # Combine signals and make decision
                    decision, confidence, strategy = self._make_trading_decision(
                        symbol, order_book_signal, technical_signal, arbitrage_signal, pump_dump_signal
                    )
                    
                    # Execute trade if decision is positive
                    if decision == 'buy' and confidence > 0.65:  # Confidence threshold
                        self._execute_trade(symbol, 'buy', confidence, strategy)
                
                except Exception as e:
                    logger.error(f"Error in trading loop for {symbol}: {e}")
            
            # Check every 5 seconds
            time.sleep(5)
    
    def _get_order_book_signal(self, symbol):
        """Generate trading signal based on order book imbalance"""
        with self.lock:
            if symbol not in self.order_book_data:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            ob_data = self.order_book_data[symbol]
            
            # Check if data is fresh (less than 5 seconds old)
            if 'computed' not in ob_data or time.time() - ob_data['computed'].get('last_update', 0) > 5:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            computed = ob_data['computed']
            imbalance = computed['imbalance']
            
            # Prepare feature vector for ML if we're using it
            if self.use_ml and symbol in self.ml_models and self.ml_models[symbol]['order_book']:
                features = [
                    imbalance,
                    computed['spread_pct'],
                    computed['depth_ratio']
                ]
                # We would need to have trained this model with actual data
                # This is just a placeholder for now
                # model = self.ml_models[symbol]['order_book']['model']
                # confidence = model.predict_proba([features])[0][1]  # Probability of class 1
                
                # For now, use a rule-based approach
                if imbalance > 0.3:  # Strong buy imbalance
                    signal = 'buy'
                    confidence = min(0.5 + imbalance/2, 0.9)  # Map imbalance to confidence
                elif imbalance < -0.3:  # Strong sell imbalance
                    signal = 'sell'
                    confidence = min(0.5 + abs(imbalance)/2, 0.9)
                else:
                    signal = 'neutral'
                    confidence = 0.5
            else:
                # Rule-based approach without ML
                if imbalance > 0.3:  # Strong buy imbalance
                    signal = 'buy'
                    confidence = min(0.5 + imbalance/2, 0.9)
                elif imbalance < -0.3:  # Strong sell imbalance
                    signal = 'sell'
                    confidence = min(0.5 + abs(imbalance)/2, 0.9)
                else:
                    signal = 'neutral'
                    confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': confidence,
                'data': {
                    'imbalance': imbalance,
                    'spread_pct': computed['spread_pct'],
                    'depth_ratio': computed['depth_ratio']
                }
            }
    
    def _get_technical_signal(self, symbol):
        """Generate trading signal based on technical indicators"""
        with self.lock:
            if symbol not in self.technical_data:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            tech_data = self.technical_data[symbol]
            data = tech_data['data']
            
            # Check if data is fresh (less than 2 minutes old)
            if time.time() - tech_data.get('timestamp', 0) > 120:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            # Use ML model if available
            if (self.use_ml and symbol in self.ml_models and 
                self.ml_models[symbol]['technical'] and 
                'model' in self.ml_models[symbol]['technical']):
                
                model_info = self.ml_models[symbol]['technical']
                model = model_info['model']
                scaler = model_info['scaler']
                features = model_info['features']
                
                # Extract feature values
                feature_values = []
                for feature in features:
                    feature_values.append(data.get(feature, 0))
                
                # Scale features
                scaled_features = scaler.transform([feature_values])
                
                # Get prediction and probability
                prediction = model.predict(scaled_features)[0]
                probabilities = model.predict_proba(scaled_features)[0]
                
                if prediction == 1:  # Buy signal
                    signal = 'buy'
                    confidence = probabilities[1]
                else:
                    signal = 'sell' if probabilities[0] > 0.7 else 'neutral'
                    confidence = probabilities[0] if signal == 'sell' else 0.5
            
            else:
                # Rule-based approach
                # Check for bullish conditions
                bullish_conditions = [
                    data.get('sma_ratio', 1) > 1.005,  # Fast SMA above slow SMA
                    data.get('macd', 0) > data.get('macd_signal', 0),  # MACD above signal
                    data.get('rsi', 50) > 50 and data.get('rsi', 50) < 70,  # RSI in bullish zone but not overbought
                    data.get('bb_position', 0.5) > 0.5 and data.get('bb_position', 0.5) < 0.9,  # Price in upper half of BB but not at extreme
                    data.get('volume_ratio', 1) > 1.2  # Higher than average volume
                ]
                
                # Check for bearish conditions
                bearish_conditions = [
                    data.get('sma_ratio', 1) < 0.995,  # Fast SMA below slow SMA
                    data.get('macd', 0) < data.get('macd_signal', 0),  # MACD below signal
                    data.get('rsi', 50) < 50 and data.get('rsi', 50) > 30,  # RSI in bearish zone but not oversold
                    data.get('bb_position', 0.5) < 0.5 and data.get('bb_position', 0.5) > 0.1,  # Price in lower half of BB but not at extreme
                    data.get('volume_ratio', 1) > 1.2  # Higher than average volume
                ]
                
                bullish_count = sum(bullish_conditions)
                bearish_count = sum(bearish_conditions)
                
                # Determine signal and confidence
                if bullish_count >= 3 and bullish_count > bearish_count + 1:
                    signal = 'buy'
                    confidence = 0.5 + (bullish_count / 10)
                elif bearish_count >= 3 and bearish_count > bullish_count + 1:
                    signal = 'sell'
                    confidence = 0.5 + (bearish_count / 10)
                else:
                    signal = 'neutral'
                    confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': confidence,
                'data': {
                    'sma_ratio': data.get('sma_ratio', 1),
                    'macd': data.get('macd', 0),
                    'rsi': data.get('rsi', 50),
                    'bb_position': data.get('bb_position', 0.5),
                    'volume_ratio': data.get('volume_ratio', 1)
                }
            }
    
    def _get_arbitrage_signal(self, symbol):
        """Generate trading signal based on arbitrage opportunities"""
        with self.lock:
            if symbol not in self.exchange_price_data or len(self.exchange_price_data[symbol]) < 2:
                return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            symbol_data = self.exchange_price_data[symbol]
            
            # Check all prices are recent (less than 30 seconds old)
            for exchange, data in symbol_data.items():
                if time.time() - data.get('timestamp', 0) > 30:
                    return {'signal': 'neutral', 'confidence': 0, 'data': {}}
            
            # Find min and max prices across exchanges
            prices = [(exchange, data['price']) for exchange, data in symbol_data.items()]
            min_exchange, min_price = min(prices, key=lambda x: x[1])
            max_exchange, max_price = max(prices, key=lambda x: x[1])
            
            # Calculate price difference percentage
            price_diff_pct = (max_price - min_price) / min_price
            
            # Factor in trading fees
            # Assume 0.1% fee per trade (0.2% round trip)
            arbitrage_threshold = 0.002  # 0.2% minimum profitable arbitrage
            
            # Determine signal and confidence
            if price_diff_pct > arbitrage_threshold:
                # Potential arbitrage opportunity
                signal = 'buy'  # Buy on cheaper exchange
                # Scale confidence based on how much the spread exceeds our threshold
                confidence = min(0.5 + (price_diff_pct - arbitrage_threshold) * 50, 0.95)
                
                # Store which exchange to buy on and which to sell on
                buy_exchange = min_exchange
                sell_exchange = max_exchange
            else:
                signal = 'neutral'
                confidence = 0.5
                buy_exchange = None
                sell_exchange = None
            
            return {
                'signal': signal,
                'confidence': confidence,
                'data': {
                    'price_diff_pct': price_diff_pct,
                    'buy_exchange': buy_exchange,
                    'sell_exchange': sell_exchange,
                    'min_price': min_price,
                    'max_price': max_price
                }
            }
