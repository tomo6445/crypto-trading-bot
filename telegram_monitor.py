"""
Telegram Pump Signal Monitor

This module monitors Telegram channels known for cryptocurrency pump signals
to provide early detection of potential pump and dump schemes. It uses the Telethon
library to connect to Telegram's API and analyze messages in real-time.

Requires:
- Telegram API credentials (api_id and api_hash)
- List of target channel names or URLs

Features:
- Connects to specified Telegram channels
- Real-time message monitoring
- Pattern recognition for pump signals
- Keyword and symbol detection
- Integration with the main trading bot
"""

import os
import re
import time
import logging
import asyncio
import json
from datetime import datetime, timedelta
from telethon import TelegramClient, events
from telethon.tl.types import Channel, User
from telethon.errors import SessionPasswordNeededError, FloodWaitError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("telegram_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("telegram_monitor")

class TelegramPumpMonitor:
    """
    Monitor Telegram channels for cryptocurrency pump signals
    """
    
    def __init__(self, api_id, api_hash, phone=None, session_file='telegram_session',
                 target_channels=None, proxy=None):
        """
        Initialize the Telegram monitor
        
        Parameters:
        - api_id: Telegram API ID
        - api_hash: Telegram API hash
        - phone: Phone number for Telegram account (optional)
        - session_file: File to store the session
        - target_channels: List of channel usernames or URLs to monitor
        - proxy: Proxy settings (optional)
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.session_file = session_file
        self.target_channels = target_channels or []
        self.proxy = proxy
        
        self.client = None
        self.running = False
        self.channel_entities = {}
        self.detected_signals = []
        self.coin_mentions = {}  # Track coin mentions over time
        
        # Pump signal detection patterns
        self.pump_keywords = [
            'pump', 'moon', 'breakout', 'surge', 'launch', 'explosion',
            'next gem', 'ready', 'alert', 'minute', 'buy now', 'buy signal',
            'targets', 'announcement', 'listing', 'exchange', 'x gain', 'huge'
        ]
        
        # Exchange names for detection
        self.exchanges = [
            'binance', 'kucoin', 'coinbase', 'huobi', 'kraken', 'bittrex',
            'ftx', 'gate.io', 'okex', 'bitfinex', 'bybit'
        ]
        
        # Create output directory
        os.makedirs('telegram_signals', exist_ok=True)
    
    async def start(self):
        """Start the Telegram monitor"""
        if self.running:
            logger.warning("Monitor already running")
            return
        
        self.running = True
        
        # Create client with proxy if provided
        if self.proxy:
            self.client = TelegramClient(
                self.session_file, self.api_id, self.api_hash,
                proxy=self.proxy
            )
        else:
            self.client = TelegramClient(self.session_file, self.api_id, self.api_hash)
        
        logger.info("Starting Telegram client...")
        
        # Connect to Telegram
        await self.client.start(phone=self.phone)
        
        # Check if we're logged in
        if not await self.client.is_user_authorized():
            logger.info("Not logged in. Please log in.")
            if self.phone:
                # Send code request
                await self.client.send_code_request(self.phone)
                try:
                    # Ask for the code
                    code = input('Enter the code you received: ')
                    await self.client.sign_in(self.phone, code)
                except SessionPasswordNeededError:
                    # 2FA is enabled
                    password = input('Enter your 2FA password: ')
                    await self.client.sign_in(password=password)
            else:
                logger.error("Phone number required for login")
                self.running = False
                return
        
        # Register event handlers
        self._register_handlers()
        
        # Process target channels
        await self._process_channels()
        
        logger.info("Telegram monitor started successfully")
    
    def _register_handlers(self):
        """Register event handlers for incoming messages"""
        @self.client.on(events.NewMessage())
        async def handle_new_message(event):
            """Handle new messages in monitored channels"""
            if not self.running:
                return
            
            try:
                # Check if message is from a monitored channel
                chat = await event.get_chat()
                chat_id = event.chat_id
                
                # Process only if from monitored channel
                if chat_id in self.channel_entities.values():
                    # Get the text
                    text = event.message.text
                    
                    # Process the message
                    await self._process_message(chat_id, chat.title if hasattr(chat, 'title') else str(chat_id), text)
            
            except Exception as e:
                logger.error(f"Error handling message: {e}")
    
    async def _process_channels(self):
        """Process the list of target channels"""
        for channel in self.target_channels:
            try:
                # Handle username or URL
                if '/' in channel:
                    # Extract username from URL
                    username = channel.split('/')[-1]
                else:
                    username = channel
                
                # Try to get entity by username
                try:
                    entity = await self.client.get_entity(username)
                    self.channel_entities[username] = entity.id
                    logger.info(f"Added channel: {username} (ID: {entity.id})")
                except Exception as e:
                    logger.error(f"Error adding channel {username}: {e}")
            
            except Exception as e:
                logger.error(f"Error processing channel {channel}: {e}")
    
    async def _process_message(self, chat_id, chat_name, text):
        """
        Process a message to detect pump signals
        
        Returns a tuple (is_pump_signal, confidence, details)
        """
        if not text or len(text.strip()) < 10:
            return False, 0, {}
        
        # Convert to lowercase for easier matching
        text_lower = text.lower()
        
        # Try to extract coin names/symbols
        coin_patterns = [
            r'\$([A-Za-z0-9]{2,10})',  # $BTC, $ETH
            r'#([A-Za-z0-9]{2,10})',   # #BTC, #ETH
            r'\b([A-Z]{2,10})/USDT\b', # BTC/USDT
            r'\b([A-Z]{2,10})-USDT\b', # BTC-USDT
            r'\b([A-Z]{2,10})USDT\b',  # BTCUSDT
            r'\b([A-Z]{2,10})/USD\b',  # BTC/USD
            r'\b([A-Z]{2,10})\b'        # BTC, ETH (more false positives)
        ]
        
        all_matches = []
        for pattern in coin_patterns:
            matches = re.findall(pattern, text)
            all_matches.extend(matches)
        
        # Remove duplicates and filter out common non-coin words
        excluded_words = ['USD', 'USDT', 'THE', 'FOR', 'AND', 'BUY', 'SELL', 'THIS', 'WITH', 'THAT', 'NOT']
        coin_symbols = [match for match in all_matches if match not in excluded_words]
        
        # Count keyword occurrences
        keyword_count = sum(1 for keyword in self.pump_keywords if keyword in text_lower)
        
        # Look for exchange mentions
        exchange_mentions = [exchange for exchange in self.exchanges if exchange in text_lower]
        
        # Look for pump indicators
        has_price_targets = bool(re.search(r'\d+%|\d+x', text_lower))
        has_timeframe = bool(re.search(r'minute|hour|today|tomorrow|shortly', text_lower))
        has_urgency = bool(re.search(r'urgent|quickly|fast|now|hurry|don\'t miss', text_lower))
        has_listing = bool(re.search(r'list(ing|ed)|launch|new coin', text_lower))
        
        # Calculate a pump signal confidence score
        confidence = 0
        
        # Base confidence from keywords
        confidence += min(0.5, keyword_count * 0.1)  # Up to 0.5 from keywords
        
        # Additional confidence from specific indicators
        if has_price_targets:
            confidence += 0.2
        
        if has_timeframe:
            confidence += 0.15
        
        if has_urgency:
            confidence += 0.15
        
        if has_listing:
            confidence += 0.1
        
        # Confidence from exchange mentions
        confidence += min(0.1, len(exchange_mentions) * 0.05)
        
        # Confidence from coin symbols
        confidence += min(0.3, len(coin_symbols) * 0.1)
        
        # Determine if this is a pump signal
        is_pump_signal = confidence >= 0.7
        
        # If we detected a signal, record it
        if is_pump_signal:
            signal_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'chat_id': chat_id,
                'chat_name': chat_name,
                'confidence': confidence,
                'coins': coin_symbols,
                'keywords': [k for k in self.pump_keywords if k in text_lower],
                'exchanges': exchange_mentions,
                'has_price_targets': has_price_targets,
                'has_timeframe': has_timeframe,
                'has_urgency': has_urgency,
                'has_listing': has_listing,
                'message': text
            }
            
            self.detected_signals.append(signal_data)
            
            # Log the signal
            logger.warning(
                f"PUMP SIGNAL DETECTED in {chat_name}! " +
                f"Confidence: {confidence:.2f}, Coins: {', '.join(coin_symbols)}"
            )
            
            # Save signal to file
            self._save_signal(signal_data)
            
            # Update coin mentions
            self._update_coin_mentions(coin_symbols)
        
        return is_pump_signal, confidence, {
            'coin_symbols': coin_symbols,
            'keyword_count': keyword_count,
            'exchange_mentions': exchange_mentions,
            'has_price_targets': has_price_targets,
            'has_timeframe': has_timeframe,
            'has_urgency': has_urgency,
            'has_listing': has_listing
        }
    
    def _save_signal(self, signal_data):
        """Save a detected signal to file"""
        try:
            # Create filename based on timestamp
            timestamp = signal_data['timestamp'].replace(':', '-').replace(' ', '_')
            filename = f"telegram_signals/signal_{timestamp}.json"
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(signal_data, f, indent=2)
            
            logger.info(f"Saved signal to {filename}")
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
    
    def _update_coin_mentions(self, coin_symbols):
        """Update coin mention tracking"""
        current_time = datetime.now()
        
        for coin in coin_symbols:
            if coin not in self.coin_mentions:
                self.coin_mentions[coin] = []
            
            self.coin_mentions[coin].append(current_time)
            
            # Clean up old mentions (older than 24 hours)
            self.coin_mentions[coin] = [
                t for t in self.coin_mentions[coin]
                if current_time - t < timedelta(hours=24)
            ]
    
    def get_trending_coins(self, min_mentions=3):
        """Get trending coins based on recent mentions"""
        current_time = datetime.now()
        trending = {}
        
        # Count recent mentions (last 2 hours)
        for coin, mentions in self.coin_mentions.items():
            recent_mentions = [
                t for t in mentions
                if current_time - t < timedelta(hours=2)
            ]
            
            if len(recent_mentions) >= min_mentions:
                trending[coin] = len(recent_mentions)
        
        # Sort by mention count
        return sorted(trending.items(), key=lambda x: x[1], reverse=True)
    
    def get_recent_signals(self, hours=2):
        """Get signals detected in the last N hours"""
        current_time = datetime.now()
        
        recent_signals = []
        for signal in self.detected_signals:
            try:
                signal_time = datetime.strptime(signal['timestamp'], '%Y-%m-%d %H:%M:%S')
                if current_time - signal_time < timedelta(hours=hours):
                    recent_signals.append(signal)
            except Exception:
                pass
        
        return recent_signals
    
    async def stop(self):
        """Stop the Telegram monitor"""
        self.running = False
        
        if self.client:
            await self.client.disconnect()
        
        logger.info("Telegram monitor stopped")

async def main():
    """Main function for standalone usage"""
    # Replace with your Telegram API credentials
    API_ID = 12345  # Your Telegram API ID
    API_HASH = "your_telegram_api_hash"
    PHONE = "+1234567890"  # Your phone number
    
    # Example target channels (known for crypto pumps)
    TARGET_CHANNELS = [
        "binance_announcements",
        "kucoin_announcements",
        "CryptoLeakGlobalNetwork",
        "CryptoSignals_News",
        "CryptoMoneyTeam",
        "CryptoBoostSignals"
    ]
    
    # Create monitor
    monitor = TelegramPumpMonitor(
        api_id=API_ID,
        api_hash=API_HASH,
        phone=PHONE,
        target_channels=TARGET_CHANNELS
    )
    
    try:
        # Start monitor
        await monitor.start()
        
        # Keep running
        print("Monitoring Telegram channels. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(60)
            
            # Print trending coins every minute
            trending = monitor.get_trending_coins()
            if trending:
                print("\nTrending coins in the last 2 hours:")
                for coin, count in trending:
                    print(f"  {coin}: {count} mentions")
            
            # Print recent signals
            recent_signals = monitor.get_recent_signals()
            if recent_signals:
                print(f"\nRecent pump signals ({len(recent_signals)}):")
                for signal in recent_signals:
                    print(f"  {signal['timestamp']} - {signal['chat_name']}")
                    print(f"    Coins: {', '.join(signal['coins'])}")
                    print(f"    Confidence: {signal['confidence']:.2f}")
    
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Stop monitor
        await monitor.stop()

if __name__ == "__main__":
    asyncio.run(main())
