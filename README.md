# crypto-trading-bot
# Advanced Cryptocurrency Trading Bot with Pump & Dump Detection

This repository contains a comprehensive algorithmic trading system designed for ultra-short-term cryptocurrency trading (30-60 minute windows). The system combines multiple advanced trading strategies with specialized pump and dump detection capabilities to capitalize on market inefficiencies and volatile price movements.

## Key Features

- **Multiple Trading Strategies**:
  - Order Book Imbalance Detection
  - Technical Indicator Momentum Scalping
  - Cross-Exchange Arbitrage
  - Statistical Arbitrage (Mean Reversion)
  - Pump and Dump Pattern Trading

- **Advanced Analysis Tools**:
  - Machine Learning signal validation
  - Social media sentiment analysis
  - Real-time Telegram channel monitoring
  - Volume anomaly detection
  - Order book microstructure analysis

- **Risk Management**:
  - Adaptive position sizing
  - Dynamic stop-losses
  - Maximum drawdown protection
  - Trade duration limits
  - Per-strategy performance tracking

## System Requirements

- Python 3.8+
- 2GB+ RAM
- Stable internet connection

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system:
   - Update `config.json` with your API keys and preferences
   - Adjust trading parameters to match your risk tolerance

## Configuration

The system is configured through the `config.json` file, which contains settings for:

- API credentials (Binance, KuCoin, etc.)
- Social media API keys (Twitter, Reddit)
- Telegram monitoring settings
- Trading parameters (amounts, targets, timeframes)
- Risk management settings
- Notification preferences
- And more...

Edit this file to match your specific requirements before running the bot.

## Usage

### Main Trading Bot

Run the main trading bot:

```bash
python trading_bot.py
```

### Pump & Dump Analyzer (Standalone)

Run the pump and dump analyzer separately:

```bash
python pump_dump_analyzer.py
```

### Telegram Monitor (Standalone)

Run the Telegram monitor separately:

```bash
python telegram_monitor.py
```

## Architecture

The system consists of multiple components that work together:

1. **Main Trading Bot** (`binance-trading-bot.py`): Core trading logic and strategy execution
2. **Pump & Dump Analyzer** (`pump-dump-analyzer.py`): Detects potential market manipulation
3. **Telegram Monitor** (`telegram-monitor.py`): Monitors crypto channels for signals

Each component can be run independently or as part of the integrated system.

## Trading Strategies Explained

### Order Book Imbalance

Monitors the ratio of buying vs. selling pressure in the order book. When significant imbalances occur (more buy orders than sell orders or vice versa), the bot enters trades in the direction of the imbalance to capitalize on potential price movements.

### Technical Indicator Momentum

Uses a combination of technical indicators (SMA, RSI, MACD, Bollinger Bands) on short timeframes to identify micro-trends. ML models enhance the accuracy of these signals by validating them against historical patterns.

### Cross-Exchange Arbitrage

Exploits price differences for the same cryptocurrency across different exchanges. When the price difference exceeds transaction costs, the bot buys on the cheaper exchange and sells on the more expensive one.

### Pump & Dump Detection

Identifies potential pump and dump schemes through:
- Abnormal volume increases
- Unusual price movements
- Social media sentiment spikes
- Telegram channel signals
- Order book anomalies

When a potential pump is detected early, the bot can enter a position to capture the upward momentum and exit before the inevitable dump phase.

## Risk Warning

**IMPORTANT**: Algorithmic trading involves significant risk, especially in volatile cryptocurrency markets. This system is provided as-is with no guarantees of profitability. Always:

- Start with small amounts
- Test thoroughly in paper trading mode first
- Monitor the system regularly
- Be prepared for potential losses
- Never invest more than you can afford to lose

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This system is built upon research from various academic papers and industry practices in high-frequency trading, market microstructure, and anomaly detection.

## Disclaimer

This software is for educational and research purposes only. The authors are not responsible for any financial losses incurred from using this system. Use at your own risk.
