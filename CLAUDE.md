# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance cryptocurrency trading strategy backtesting and optimization framework with GPU acceleration. Supports RSI, MACD, and EMA strategies with multiple optimization approaches.

## Essential Commands

### Setup & Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify GPU availability (optional)
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Automated installation (WSL/Linux)
bash install.sh
```

### Running Backtests

```bash
# TensorFlow GPU optimization (FASTEST - 15-20 tests/second)
python tests/backtest_runner.py --rsi --tensorflow-gpu
python tests/backtest_runner.py --macd --tensorflow-gpu
python tests/backtest_runner.py --ema --tensorflow-gpu
python tests/backtest_runner.py --all-tensorflow-gpu

# Basic grid search (systematic parameter exploration)
python tests/backtest_runner.py --rsi --basic-strategy
python tests/backtest_runner.py --macd --basic-strategy
python tests/backtest_runner.py --ema --basic-strategy
python tests/backtest_runner.py --all --basic-strategy

# Random search (RSI only - stochastic sampling)
python tests/backtest_runner.py --rsi --random-search

# Custom iterations
python tests/backtest_runner.py --rsi --tensorflow-gpu --iterations 5000
```

### Data Management

```bash
# Fetch new data from Binance
python data/fetch_binance_data.py

# Available data: BTC/USDT, ETH/USDT, SOL/USDT
# Timeframes: 5m, 15m, 30m, 1h, 4h, 1d
```

### Code Quality

```bash
# Format code
black . --line-length 120
isort .
flake8 . --max-line-length=120
```

## Architecture

### Core Components

1. **Universal Test Runner** (`tests/backtest_runner.py`)
   - Single entry point for all strategies and optimization modes
   - Automatic dependency checking with installation guidance
   - GPU detection with CPU fallback
   - Performance comparison reports across strategies

2. **Strategy Implementation Pattern**
   Each strategy follows consistent structure:
   - **Parameters Class**: Dataclass with validation (e.g., `RSIParameters`)
   - **Strategy Class**: Core with `backtest()` and `optimize_parameters()` methods
   - **Signal Enum**: BUY=1, SELL=-1, HOLD=0
   - **Optimizer**: GPU-accelerated parameter testing

3. **Optimization Methods**
   - **TensorFlow GPU**: Hybrid GPU/CPU approach (15-20 tests/sec)
   - **Simple GPU**: Vectorized calculations (700+ tests/sec for RSI)
   - **Grid Search**: Systematic parameter exploration
   - **Random Search**: Stochastic sampling (RSI only)

### Strategy Interfaces

All strategies implement:
```python
def backtest(data: pd.DataFrame, params: Parameters) -> BacktestResult
def optimize_parameters(data: pd.DataFrame, mode: str) -> OptimizationResult
```

**RSI Strategy**: Buy oversold (<30), sell overbought (>70)
**MACD Strategy**: Trade on MACD/signal line crossovers
**EMA Strategy**: Trade on fast/slow EMA crossovers

### Key Technical Details

- **Random Sampling**: Uses random 1-year data segments to prevent overfitting
- **Batch Processing**: Efficient parameter testing with progress tracking
- **GPU Fallback**: Automatic CPU mode if GPU unavailable
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, total return
- **Data Format**: CSV with OHLCV columns from Binance Futures

### Important Files

- `tests/backtest_runner.py` - Main entry point with all command-line options
- `strategies/*/[strategy]_strategy.py` - Core strategy implementations
- `strategies/*/[strategy]_optimizer.py` - GPU optimization logic
- `data/fetch_binance_data.py` - Data fetching and updates
- `requirements.txt` - Python dependencies including TensorFlow and CuPy