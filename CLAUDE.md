# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance cryptocurrency trading strategy backtesting and optimization framework with GPU acceleration. Supports RSI, MACD, and EMA strategies with multiple optimization approaches.

## Quick Start

```bash
# Check if virtual environment exists, activate or create
if [ -d ".venv_wsl" ]; then source .venv_wsl/bin/activate; else python3 -m venv .venv_wsl && source .venv_wsl/bin/activate; fi

# Install dependencies
pip install -r requirements.txt

# Run a quick test with RSI strategy
python tests/backtest_runner.py --rsi --basic-strategy
```

## Essential Commands

### Setup & Dependencies

```bash
# Virtual environment management
source .venv_wsl/bin/activate  # Activate existing environment
python3 -m venv .venv_wsl      # Create new environment if needed

# Install all dependencies
pip install -r requirements.txt

# Verify GPU availability (optional)
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Automated installation with GPU support (WSL/Linux)
bash install.sh  # Handles CUDA, cuDNN, TensorFlow GPU setup
```

### Running Backtests

```bash
# TRUE GPU PARALLEL (ULTRA FAST - 2500+ tests/second)
python tests/backtest_runner.py --rsi --tensorflow-gpu   # ~8 seconds for 20K params
python tests/backtest_runner.py --macd --tensorflow-gpu  # ~3 seconds for 1K params
python tests/backtest_runner.py --ema --tensorflow-gpu   # ~2 seconds for 500 params
python tests/backtest_runner.py --all-tensorflow-gpu     # All strategies sequentially

# Basic grid search (CPU - slower but universal)
python tests/backtest_runner.py --rsi --basic-strategy
python tests/backtest_runner.py --macd --basic-strategy
python tests/backtest_runner.py --ema --basic-strategy
python tests/backtest_runner.py --all --basic-strategy

# Random search (RSI only - stochastic sampling)
python tests/backtest_runner.py --rsi --random-search
```

### Data Management

```bash
# Fetch new data from Binance
python data/fetch_binance_data.py

# Available data files in data/:
# - btcusdt_5m.csv, btcusdt_15m.csv, btcusdt_30m.csv
# - btcusdt_1h.csv, btcusdt_4h.csv, btcusdt_1d.csv
# - ethusdt_*.csv, solusdt_*.csv (same timeframes)

# Data format: CSV with columns:
# timestamp, open, high, low, close, volume
```

### Code Quality & Testing

```bash
# Format code
black . --line-length 120
isort .
flake8 . --max-line-length=120

# Run tests
pytest tests/  # If tests exist

# Quick validation - run each strategy
python tests/backtest_runner.py --rsi --basic-strategy
python tests/backtest_runner.py --macd --basic-strategy
python tests/backtest_runner.py --ema --basic-strategy
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
   - **TensorFlow GPU**: TRUE parallel GPU execution (2500+ tests/sec)
   - **Grid Search**: Systematic parameter exploration (CPU)
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

- **TRUE GPU Parallelism**: All parameter combinations tested simultaneously
- **Vectorized Operations**: No for-loops, pure tensor operations
- **XLA JIT Compilation**: Maximum GPU performance with TensorFlow
- **Batch Processing**: Memory-efficient processing in chunks
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, total return
- **Data Format**: CSV with OHLCV columns from Binance Futures

### Performance Benchmarks

| Strategy | Parameters | CPU Time | GPU Time | Speedup |
|----------|------------|----------|----------|---------|
| RSI | 20,286 | ~6 hours | 8 seconds | 2,700x |
| MACD | 1,320 | ~30 min | 3 seconds | 600x |
| EMA | 88 | ~3 min | 1 second | 180x |

### Project Structure

```
.
├── strategies/             # Trading strategy implementations
│   ├── rsi/
│   │   ├── rsi_strategy.py         # Core RSI logic
│   │   └── rsi_gpu_optimizer.py    # GPU-accelerated optimization
│   ├── macd/
│   │   ├── macd_strategy.py        # Core MACD logic
│   │   └── macd_gpu_optimizer.py   # GPU-accelerated optimization
│   └── ema/
│       ├── ema_strategy.py         # Core EMA logic
│       └── ema_gpu_optimizer.py    # GPU-accelerated optimization
├── tests/
│   └── backtest_runner.py  # Universal entry point for all strategies
├── data/                    # Market data (CSV files)
│   └── fetch_binance_data.py  # Data fetching utility
├── results/                 # Optimization results (created on run)
├── config.json             # Strategy parameters and ranges
├── requirements.txt        # Python dependencies
└── install.sh             # Automated GPU setup script
```

### Configuration (config.json)

- **strategies**: Parameter ranges and defaults for RSI, MACD, EMA
- **backtest_settings**: Initial capital, commission, slippage settings
- **optimization_settings**: GPU batch size, mixed precision, XLA settings
- **data_settings**: Default data file, available pairs and timeframes

### Command Line Arguments

`tests/backtest_runner.py` accepts:
- Strategy selection: `--rsi`, `--macd`, `--ema`, `--all`
- Optimization mode: `--tensorflow-gpu`, `--basic-strategy`, `--random-search`
- Combined: `--all-tensorflow-gpu` (all strategies with GPU)

### Results Output

Optimization results are saved to `results/` directory:
- CSV files with parameter combinations and metrics
- Performance statistics (Sharpe ratio, returns, drawdown)
- Execution time benchmarks