# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A high-performance trading strategy backtesting and optimization framework with GPU acceleration. Supports RSI, MACD, and EMA strategies with multiple optimization approaches including grid search, random search, and TensorFlow GPU optimization.

## Essential Commands

### Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# For GPU support, ensure NVIDIA CUDA Toolkit 12.x is installed
nvidia-smi  # Verify GPU availability
```

### Running Backtests

```bash
# TensorFlow GPU optimization (RECOMMENDED - fastest)
python tests/backtest_runner.py --rsi --tensorflow-gpu
python tests/backtest_runner.py --macd --tensorflow-gpu
python tests/backtest_runner.py --ema --tensorflow-gpu

# Run all strategies with GPU optimization sequentially
python tests/backtest_runner.py --all-tensorflow-gpu

# Basic strategy with grid search
python tests/backtest_runner.py --rsi --basic-strategy
python tests/backtest_runner.py --macd --basic-strategy
python tests/backtest_runner.py --ema --basic-strategy

# Random search optimization (RSI only)
python tests/backtest_runner.py --rsi --random-search

# Run all strategies
python tests/backtest_runner.py --all --basic-strategy
```

### Code Quality

```bash
# Install formatting tools
pip install black isort flake8

# Format code
black strategies/ tests/ data/ --line-length 120
isort strategies/ tests/ data/
flake8 strategies/ tests/ data/ --max-line-length=120
```

## Architecture

### Core Components

1. **Test Runner** (`tests/backtest_runner.py`)
   - Entry point for all strategies
   - Handles data loading, optimization mode selection, report generation
   - Supports multiple optimization modes via command-line flags
   - Checks dependencies and provides installation instructions

2. **Strategies** (`strategies/`)
   - Each strategy in its own directory: `rsi/`, `macd/`, `ema/`
   - Consistent structure per strategy:
     - `{strategy}_strategy.py` - Core implementation with backtest logic
     - `{strategy}_optimizer.py` - GPU/CPU optimization logic
     - `gpu_optimizer_svg_report.py` - SVG report generation

3. **Data** (`data/`)
   - Pre-fetched CSV files: BTC, ETH, SOL pairs
   - Timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d
   - `fetch_binance_data.py` - Binance data fetching utilities

4. **Reports** (`reports/`)
   - Generated SVG reports with interactive visualizations
   - Performance metrics, parameter distributions, optimization results

### Optimization Approaches

- **TensorFlow GPU**: Hybrid approach using GPU for parameter generation, CPU for backtesting
- **Random Search**: Stochastic parameter exploration
- **Grid Search**: Systematic parameter space exploration

### Key Technical Details

- GPU optimization uses TensorFlow with automatic GPU detection and CPU fallback
- Random 1-year data segments for each test to prevent overfitting
- Batch processing of parameter combinations for efficiency
- All strategies implement consistent interface: `backtest()` and `optimize_parameters()`