"""
Advanced Pump and Dump Pattern Detection and Analysis

This module extends the trading bot with specialized algorithms to detect
potential pump and dump events in cryptocurrency markets using various sources:
1. Volume anomaly detection
2. Price movement patterns
3. Order book analysis
4. Social media monitoring
5. Market microstructure analysis

Can be used as a standalone analyzer or integrated with the main trading bot.
"""

import pandas as pd
import numpy as np
import time
import math
import logging
import datetime
from binance.client import Client
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import requests
import json
from collections import defaultdict, Counter
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pump_dump_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pump_dump_analyzer")

class PumpDumpAnalyzer:
    """
    Advanced analyzer for detecting pump and dump patterns in cryptocurrency markets
    """
    
    def __init__(self, api_key, api_secret, symbols=None, social_api_keys=None, 
                 telegram_channels=None, lookback_hours=24):
        """
        Initialize the pump and dump analyzer
        
        Parameters:
        - api_key: Binance API key
        - api_secret: Binance API secret
        - symbols: List of trading pairs to monitor (default: top 30 by volume)
        - social_api_keys: API keys for social media platforms
        - telegram_channels: List of Telegram channels to monitor
        - lookback_hours: How many hours of historical data to consider
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = Client(api_key, api_secret)
        self.lookback_hours = lookback_hours
        self.telegram_channels = telegram_channels or []
        self.social_api_keys = social_api_keys or {}
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            logger.warning("Failed to initialize sentiment analyzer")
            self.sentiment_analyzer = None
        
        # State variables
        self.running = False
        self.lock = threading.Lock()
        self.anomaly_models = {}
        self.historical_data = {}
        self.pump_dump_alerts = []
        self.market_state = {}
        self.social_sentiment = {}
        
        # Determine symbols to analyze
        if symbols:
            self.symbols = symbols
        else:
            # Get top 30 USDT pairs by volume
            tickers = self.client.get_ticker()
            usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
            usdt_pairs.sort(key=lambda x: float(x['volume']), reverse=True)
            self.symbols = [p['symbol'] for p in usdt_pairs[:30]]
        
        logger.info(f"Pump & Dump Analyzer initialized with {len(self.symbols)} symbols")
    
    def start(self):
        """Start the analyzer"""
        if self.running:
            logger.warning("Analyzer already running")
            return
        
        self.running = True
        
        # Start data collection threads
        threading.Thread(target=self._collect_historical_data, daemon=True).start()
        threading.Thread(target=self._monitor_live_markets, daemon=True).start()
        threading.Thread(target=self._collect_social_data, daemon=True).start()
        threading.Thread(target=self._analyze_patterns, daemon=True).start()
        
        logger.info("Pump & Dump Analyzer started")
    
    def stop(self):
        """Stop the analyzer"""
        self.running = False
        logger.info("Pump & Dump Analyzer stopped")
    
    def _collect_historical_data(self):
        """Collect historical price and volume data for analysis"""
        # Wait a bit to avoid hitting API limits immediately
        time.sleep(5)
        
        while self.running:
            for symbol in self.symbols:
                try:
                    # Get historical klines data
                    end_time = int(time.time() * 1000)
                    start_time = end_time - (self.lookback_hours * 60 * 60 * 1000)
                    
                    # Get 1-minute klines
                    klines = self.client.get_historical_klines(
                        symbol, Client.KLINE_INTERVAL_1MINUTE,
                        start_str=start_time, end_str=end_time
                    )
                    
                    # Convert to dataframe
                    if not klines:
                        logger.warning(f"No historical data available for {symbol}")
                        continue
                        
                    df = pd.DataFrame(klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert types
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Convert timestamps to datetime
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                    
                    # Calculate additional features
                    df['price_pct_change'] = df['close'].pct_change() * 100
                    df['volume_pct_change'] = df['volume'].pct_change() * 100
                    
                    # Calculate moving averages
                    df['volume_ma20'] = df['volume'].rolling(20).mean()
                    df['price_ma20'] = df['close'].rolling(20).mean()
                    df['volume_ratio'] = df['volume'] / df['volume_ma20']
                    
                    # Calculate volatility
                    df['price_std20'] = df['close'].rolling(20).std()
                    df['price_volatility'] = df['price_std20'] / df['price_ma20']
                    
                    # Calculate RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    
                    # Store the data
                    with self.lock:
                        self.historical_data[symbol] = df
                    
                    # Train anomaly detection model
                    self._train_anomaly_model(symbol, df)
                    
                    logger.debug(f"Collected historical data for {symbol}: {len(df)} points")
                    
                except Exception as e:
                    logger.error(f"Error collecting historical data for {symbol}: {e}")
            
            # Update every 15 minutes
            time.sleep(15 * 60)
    
    def _train_anomaly_model(self, symbol, df):
        """Train anomaly detection model for volume and price patterns"""
        try:
            # Drop rows with NaN values
            df_clean = df.dropna()
            
            if len(df_clean) < 100:
                logger.warning(f"Not enough data to train anomaly model for {symbol}")
                return
            
            # Select features for anomaly detection
            features = [
                'volume_ratio', 'price_pct_change', 'price_volatility', 'rsi'
            ]
            
            # Check if all features are available
            if not all(f in df_clean.columns for f in features):
                missing = [f for f in features if f not in df_clean.columns]
                logger.warning(f"Missing features for {symbol}: {missing}")
                return
            
            X = df_clean[features].values
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Isolation Forest model
            model = IsolationForest(
                n_estimators=100,
                contamination=0.05,  # Expect 5% of data to be anomalies
                random_state=42
            )
            model.fit(X_scaled)
            
            # Store model and scaler
            with self.lock:
                self.anomaly_models[symbol] = {
                    'model': model,
                    'scaler': scaler,
                    'features': features
                }
            
            logger.info(f"Trained anomaly detection model for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training anomaly model for {symbol}: {e}")
    
    def _monitor_live_markets(self):
        """Monitor live market data for real-time analysis"""
        while self.running:
            for symbol in self.symbols:
                try:
                    # Get current ticker data
                    ticker = self.client.get_ticker(symbol=symbol)
                    
                    # Get recent trades
                    trades = self.client.get_recent_trades(symbol=symbol, limit=100)
                    
                    # Get order book
                    depth = self.client.get_order_book(symbol=symbol, limit=20)
                    
                    # Process the data
                    with self.lock:
                        self.market_state[symbol] = {
                            'ticker': ticker,
                            'last_price': float(ticker['lastPrice']),
                            'price_change_pct': float(ticker['priceChangePercent']),
                            '24h_volume': float(ticker['volume']),
                            'trades': trades,
                            'order_book': depth,
                            'timestamp': time.time()
                        }
                        
                        # Calculate order book imbalance
                        bid_volume = sum(float(bid[1]) for bid in depth['bids'])
                        ask_volume = sum(float(ask[1]) for ask in depth['asks'])
                        
                        if bid_volume + ask_volume > 0:
                            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                        else:
                            imbalance = 0
                            
                        self.market_state[symbol]['order_book_imbalance'] = imbalance
                
                except Exception as e:
                    logger.error(f"Error monitoring live market for {symbol}: {e}")
            
            # Update every 10 seconds
            time.sleep(10)
    
    def _collect_social_data(self):
        """Collect social media mentions and sentiment data"""
        if not self.social_api_keys and not self.telegram_channels:
            logger.warning("No social API keys or Telegram channels provided")
            return
        
        while self.running:
            # Extract base symbols from trading pairs (e.g., BTC from BTCUSDT)
            base_symbols = [re.sub(r'USDT$|BTC$|ETH$', '', symbol) for symbol in self.symbols]
            base_symbols = [s for s in base_symbols if s]  # Remove empty strings
            
            # Monitor social media mentions
            try:
                # This would call Twitter API, Reddit API, etc.
                # Placeholder for now
                social_data = self._get_mock_social_data(base_symbols)
                
                # Update social sentiment data
                with self.lock:
                    for symbol, data in social_data.items():
                        trading_symbol = f"{symbol}USDT"  # Convert back to trading pair format
                        if trading_symbol in self.symbols:
                            self.social_sentiment[trading_symbol] = {
                                'data': data,
                                'timestamp': time.time()
                            }
            
            except Exception as e:
                logger.error(f"Error collecting social data: {e}")
            
            # Check every 5 minutes
            time.sleep(5 * 60)
    
    def _get_mock_social_data(self, symbols):
        """Mock function for social media data (replace with actual API calls)"""
        result = {}
        
        for symbol in symbols:
            # Generate mock data
            result[symbol] = {
                'mentions': np.random.randint(0, 100),
                'sentiment_score': np.random.uniform(-1, 1),
                'source': 'mock_data'
            }
        
        return result
    
    def _analyze_patterns(self):
        """Main analysis loop to detect pump and dump patterns"""
        # Wait for initial data collection
        time.sleep(60)
        
        while self.running:
            for symbol in self.symbols:
                try:
                    if symbol not in self.historical_data or symbol not in self.market_state:
                        continue
                    
                    # Get historical and current data
                    with self.lock:
                        hist_data = self.historical_data[symbol].copy()
                        market_data = self.market_state[symbol].copy()
                        model_info = self.anomaly_models.get(symbol)
                        social_data = self.social_sentiment.get(symbol, {}).get('data', {})
                    
                    if not model_info:
                        continue
                    
                    # Run anomaly detection
                    is_anomaly, anomaly_score, details = self._detect_anomalies(
                        symbol, hist_data, market_data, model_info, social_data
                    )
                    
                    if is_anomaly:
                        # Create an alert
                        alert = {
                            'symbol': symbol,
                            'timestamp': time.time(),
                            'score': anomaly_score,
                            'type': details['type'],
                            'confidence': details['confidence'],
                            'indicators': details['indicators'],
                            'description': details['description'],
                            'action': details['action']
                        }
                        
                        # Store the alert
                        with self.lock:
                            self.pump_dump_alerts.append(alert)
                        
                        # Log the alert
                        logger.warning(
                            f"ALERT: {details['type']} detected for {symbol} " +
                            f"(confidence: {details['confidence']:.2f}). " +
                            f"Action: {details['action']}"
                        )
                
                except Exception as e:
                    logger.error(f"Error analyzing patterns for {symbol}: {e}")
            
            # Check every 30 seconds
            time.sleep(30)
    
    def _detect_anomalies(self, symbol, hist_data, market_data, model_info, social_data):
        """
        Detect anomalies in trading data that might indicate pump and dump patterns
        
        Returns:
        - is_anomaly: Boolean indicating if an anomaly was detected
        - anomaly_score: Score indicating the severity of the anomaly
        - details: Dictionary with details about the anomaly
        """
        # Initialize result
        is_anomaly = False
        anomaly_score = 0
        details = {
            'type': 'none',
            'confidence': 0,
            'indicators': [],
            'description': '',
            'action': 'none'
        }
        
        # Skip if data is insufficient
        if len(hist_data) < 20:
            return is_anomaly, anomaly_score, details
        
        # Get recent data rows
        recent_data = hist_data.tail(20)
        
        # Collect detected signals
        pump_signals = []
        dump_signals = []
        
        # 1. Check for volume spikes
        recent_volume = recent_data['volume'].iloc[-1]
        avg_volume = recent_data['volume'].iloc[:-1].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 5:  # Volume 5x higher than recent average
            pump_signals.append({
                'indicator': 'volume_spike',
                'value': volume_ratio,
                'threshold': 5,
                'weight': 0.8
            })
        
        # 2. Check for rapid price increases
        recent_price_change = recent_data['price_pct_change'].iloc[-1]
        if recent_price_change > 5:  # 5% price increase in 1 minute
            pump_signals.append({
                'indicator': 'price_spike',
                'value': recent_price_change,
                'threshold': 5,
                'weight': 0.9
            })
        
        # 3. Check for order book imbalance
        order_book_imbalance = market_data.get('order_book_imbalance', 0)
        if order_book_imbalance > 0.5:  # Strong buy imbalance
            pump_signals.append({
                'indicator': 'buy_imbalance',
                'value': order_book_imbalance,
                'threshold': 0.5,
                'weight': 0.7
            })
        elif order_book_imbalance < -0.5:  # Strong sell imbalance
            dump_signals.append({
                'indicator': 'sell_imbalance',
                'value': order_book_imbalance,
                'threshold': -0.5,
                'weight': 0.7
            })
        
        # 4. Check for RSI extremes
        recent_rsi = recent_data['rsi'].iloc[-1]
        if not math.isnan(recent_rsi):
            if recent_rsi > 85:  # Extremely overbought
                dump_signals.append({
                    'indicator': 'overbought_rsi',
                    'value': recent_rsi,
                    'threshold': 85,
                    'weight': 0.6
                })
            elif recent_rsi < 20:  # Extremely oversold (could precede a pump)
                pump_signals.append({
                    'indicator': 'oversold_rsi',
                    'value': recent_rsi,
                    'threshold': 20,
                    'weight': 0.4
                })
        
        # 5. Check social media signals
        if social_data:
            mentions = social_data.get('mentions', 0)
            sentiment = social_data.get('sentiment_score', 0)
            
            if mentions > 50 and sentiment > 0.5:  # High positive mentions
                pump_signals.append({
                    'indicator': 'social_hype',
                    'value': mentions * sentiment,
                    'threshold': 25,  # 50 mentions * 0.5 sentiment
                    'weight': 0.5
                })
        
        # 6. Use the trained anomaly detection model
        if model_info and 'model' in model_info and 'scaler' in model_info:
            try:
                model = model_info['model']
                scaler = model_info['scaler']
                features = model_info['features']
                
                # Create feature vector for current data
                current_features = []
                for feature in features:
                    if feature in recent_data.columns:
                        current_features.append(recent_data[feature].iloc[-1])
                    else:
                        # Use a default value or skip this feature
                        current_features.append(0)
                
                current_features = np.array([current_features])
                current_features_scaled = scaler.transform(current_features)
                
                # Predict anomaly
                prediction = model.predict(current_features_scaled)[0]
                anomaly_score = model.decision_function(current_features_scaled)[0]
                
                # Negative prediction means anomaly
                if prediction == -1:
                    # Determine if it's likely a pump or dump
                    if recent_price_change > 0:
                        pump_signals.append({
                            'indicator': 'anomaly_detection',
                            'value': anomaly_score,
                            'threshold': 0,
                            'weight': 0.75
                        })
                    else:
                        dump_signals.append({
                            'indicator': 'anomaly_detection',
                            'value': anomaly_score,
                            'threshold': 0,
                            'weight': 0.75
                        })
            except Exception as e:
                logger.error(f"Error using anomaly model for {symbol}: {e}")
        
        # Synthesize all signals to determine overall pattern
        if pump_signals:
            pump_weight_sum = sum(s['weight'] for s in pump_signals)
            pump_score = sum(s['weight'] * min(abs(s['value'] / s['threshold']), 5) for s in pump_signals) / pump_weight_sum
            pump_confidence = min(0.95, pump_score * 0.9)  # Cap at 0.95
        else:
            pump_confidence = 0
        
        if dump_signals:
            dump_weight_sum = sum(s['weight'] for s in dump_signals)
            dump_score = sum(s['weight'] * min(abs(s['value'] / s['threshold']), 5) for s in dump_signals) / dump_weight_sum
            dump_confidence = min(0.95, dump_score * 0.9)  # Cap at 0.95
        else:
            dump_confidence = 0
        
        # Determine final output
        if pump_signals and pump_confidence > 0.7:
            is_anomaly = True
            anomaly_score = pump_confidence
            details = {
                'type': 'pump_pattern',
                'confidence': pump_confidence,
                'indicators': [s['indicator'] for s in pump_signals],
                'description': f"Potential pump pattern detected in {symbol}",
                'action': 'buy_opportunity' if pump_confidence > 0.8 else 'monitor'
            }
        elif dump_signals and dump_confidence > 0.7:
            is_anomaly = True
            anomaly_score = dump_confidence
            details = {
                'type': 'dump_pattern',
                'confidence': dump_confidence,
                'indicators': [s['indicator'] for s in dump_signals],
                'description': f"Potential dump pattern detected in {symbol}",
                'action': 'sell_immediately' if dump_confidence > 0.8 else 'prepare_exit'
            }
        
        return is_anomaly, anomaly_score, details
    
    def get_active_alerts(self, min_confidence=0.7, max_age_minutes=30):
        """Get active alerts that meet the confidence threshold and are recent"""
        current_time = time.time()
        
        with self.lock:
            active_alerts = [
                alert for alert in self.pump_dump_alerts
                if alert['confidence'] >= min_confidence
                and (current_time - alert['timestamp']) <= (max_age_minutes * 60)
            ]
        
        return active_alerts
    
    def get_alert_for_symbol(self, symbol):
        """Get the most recent alert for a specific symbol"""
        with self.lock:
            symbol_alerts = [a for a in self.pump_dump_alerts if a['symbol'] == symbol]
            if not symbol_alerts:
                return None
            
            # Return most recent alert
            return max(symbol_alerts, key=lambda x: x['timestamp'])
    
    def generate_report(self, output_file=None):
        """Generate a report of detected pump and dump patterns"""
        # Get current time
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get active alerts
        active_alerts = self.get_active_alerts()
        
        # Start building the report
        report = f"Pump and Dump Pattern Analysis Report\n"
        report += f"Generated: {current_time}\n"
        report += f"Monitoring {len(self.symbols)} symbols\n\n"
        
        # Add active alerts
        report += f"Active Alerts: {len(active_alerts)}\n"
        report += "-" * 80 + "\n"
        
        for alert in active_alerts:
            report += f"Symbol: {alert['symbol']}\n"
            report += f"Type: {alert['type']}\n"
            report += f"Confidence: {alert['confidence']:.2f}\n"
            report += f"Time: {datetime.datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"Indicators: {', '.join(alert['indicators'])}\n"
            report += f"Action: {alert['action']}\n"
            report += "-" * 80 + "\n"
        
        # If no active alerts
        if not active_alerts:
            report += "No active alerts at this time.\n"
            report += "-" * 80 + "\n"
        
        # Add summary of monitored symbols
        report += "\nMarket Overview:\n"
        report += "-" * 80 + "\n"
        
        with self.lock:
            for symbol in sorted(self.market_state.keys()):
                data = self.market_state[symbol]
                report += f"{symbol}: {data.get('last_price', 'N/A')}, "
                report += f"24h Change: {data.get('price_change_pct', 'N/A')}%, "
                report += f"Order Book Balance: {data.get('order_book_imbalance', 'N/A'):.2f}\n"
        
        # Write to file if specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report)
                logger.info(f"Report written to {output_file}")
            except Exception as e:
                logger.error(f"Error writing report to {output_file}: {e}")
        
        return report

# Example usage
if __name__ == "__main__":
    # Replace with your API credentials
    API_KEY = "your_binance_api_key"
    API_SECRET = "your_binance_api_secret"
    
    # Create analyzer
    analyzer = PumpDumpAnalyzer(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'SHIBUSDT', 'SOLUSDT'],
        lookback_hours=48
    )
    
    # Start analyzer
    try:
        analyzer.start()
        
        # Run for a specific amount of time
        running_time = 60 * 60 * 2  # 2 hours
        print(f"Analyzer started. Will run for {running_time/3600} hours.")
        
        # Check for alerts periodically
        end_time = time.time() + running_time
        while time.time() < end_time:
            time.sleep(300)  # Check every 5 minutes
            
            # Get active alerts
            alerts = analyzer.get_active_alerts()
            if alerts:
                print(f"\nActive Alerts ({len(alerts)}):")
                for alert in alerts:
                    alert_time = datetime.datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
                    print(f"{alert_time} - {alert['symbol']}: {alert['type']} (Confidence: {alert['confidence']:.2f})")
                    print(f"  Action: {alert['action']}")
                    print(f"  Indicators: {', '.join(alert['indicators'])}")
            
            # Generate report every hour
            if int((end_time - time.time()) / 60) % 60 == 0:
                report = analyzer.generate_report(output_file="pump_dump_report.txt")
                print("\nGenerated report: pump_dump_report.txt")
        
        # Shutdown gracefully
        analyzer.stop()
        print("Analyzer stopped.")
        
    except KeyboardInterrupt:
        print("\nStopping analyzer due to user interrupt...")
        analyzer.stop()
        
        # Generate final report
        report = analyzer.generate_report(output_file="pump_dump_final_report.txt")
        print("\nGenerated final report: pump_dump_final_report.txt")
    
    except Exception as e:
        print(f"Error running analyzer: {e}")
        if 'analyzer' in locals() and analyzer.running:
            analyzer.stop()
