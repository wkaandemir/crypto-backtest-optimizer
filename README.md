# Crypto Backtest Optimizer

A high-performance cryptocurrency trading strategy backtesting and optimization framework with GPU acceleration support.

## Features

- **Multiple Trading Strategies**: RSI, MACD, and EMA indicators
- **GPU Acceleration**: TensorFlow GPU optimization for ultra-fast parameter testing
- **Multiple Optimization Methods**: Grid search, random search, and GPU-accelerated optimization
- **Comprehensive Reporting**: Interactive SVG reports with performance metrics
- **Multi-Timeframe Support**: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- **Multiple Trading Pairs**: BTC/USDT, ETH/USDT, SOL/USDT

## Installation

### Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- CUDA Toolkit 12.x (for GPU features)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-backtest-optimizer.git
cd crypto-backtest-optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify GPU availability (optional):
```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Quick Start

### Running Backtests

#### GPU Optimization (Fastest)
```bash
# Test individual strategies
python tests/backtest_runner.py --rsi --tensorflow-gpu
python tests/backtest_runner.py --macd --tensorflow-gpu
python tests/backtest_runner.py --ema --tensorflow-gpu

# Run all strategies with GPU
python tests/backtest_runner.py --all-tensorflow-gpu
```

#### Grid Search
```bash
# Basic strategy testing
python tests/backtest_runner.py --rsi --basic-strategy
python tests/backtest_runner.py --macd --basic-strategy
python tests/backtest_runner.py --ema --basic-strategy

# All strategies
python tests/backtest_runner.py --all --basic-strategy
```

#### Random Search
```bash
# RSI with random search
python tests/backtest_runner.py --rsi --random-search
```

## Project Structure

```
crypto-backtest-optimizer/
├── strategies/           # Trading strategy implementations
│   ├── rsi/             # RSI strategy
│   ├── macd/            # MACD strategy
│   └── ema/             # EMA crossover strategy
├── data/                # Historical price data
│   └── *.csv            # Pre-fetched OHLCV data
├── tests/               # Test runner and utilities
│   └── backtest_runner.py
├── reports/             # Generated backtest reports
└── requirements.txt     # Python dependencies
```

## Strategies

### RSI (Relative Strength Index)
- Identifies overbought/oversold conditions
- Parameters: RSI period, oversold/overbought thresholds
- Best for: Range-bound markets

### MACD (Moving Average Convergence Divergence)
- Trend-following momentum indicator
- Parameters: Fast/slow/signal periods
- Best for: Trending markets

### EMA (Exponential Moving Average)
- Crossover strategy with dual EMAs
- Parameters: Short/long periods
- Best for: Trend identification

## Performance Optimization

### GPU Acceleration
The framework uses TensorFlow for GPU-accelerated parameter generation:
- 100x+ faster than CPU-only optimization
- Automatic fallback to CPU if GPU unavailable
- Batch processing for maximum efficiency

### Optimization Methods

1. **Grid Search**: Systematic exploration of parameter space
2. **Random Search**: Stochastic sampling with broader coverage
3. **TensorFlow GPU**: Hybrid GPU/CPU approach for maximum speed

## Reports

Generated reports include:
- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Equity curves and drawdown visualization
- Parameter distribution analysis
- Trade statistics and monthly returns
- Best parameter combinations

Reports are saved as interactive SVG files in the `reports/` directory.

## Data

Historical data includes:
- **Pairs**: BTC/USDT, ETH/USDT, SOL/USDT
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- **Source**: Binance exchange
- **Format**: CSV with OHLCV data

To fetch new data:
```python
from data.fetch_binance_data import fetch_and_save_data
fetch_and_save_data('BTCUSDT', '1h', days=365)
```

## Development

### Code Formatting
```bash
# Install formatting tools
pip install black isort flake8

# Format code
black . --line-length 120
isort .
flake8 . --max-line-length=120
```

### Running Tests
```bash
# Run specific strategy test
python tests/backtest_runner.py --rsi --basic-strategy

# Run with custom parameters
python tests/backtest_runner.py --macd --tensorflow-gpu --iterations 5000
```

## Performance Tips

1. **Use GPU optimization** for fastest results
2. **Start with smaller datasets** (5m, 15m) for initial testing
3. **Use random search** for initial parameter exploration
4. **Use grid search** for fine-tuning around promising parameters
5. **Monitor GPU memory** usage with `nvidia-smi`

## Requirements

Key dependencies:
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `ta`: Technical indicators
- `tensorflow`: GPU acceleration
- `plotly`: Interactive visualizations
- `ccxt`: Exchange data fetching

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.