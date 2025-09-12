"""
MACD (Moving Average Convergence Divergence) Trading Strategy

This module implements a MACD-based trading strategy with configurable parameters
for backtesting and optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """Trading signals"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class MACDParameters:
    """MACD strategy parameters"""
    fast_period: int = 12  # Fast EMA period
    slow_period: int = 26  # Slow EMA period
    signal_period: int = 9  # Signal line EMA period
    
    def validate(self):
        """Validate parameter ranges"""
        if self.fast_period <= 0 or self.slow_period <= 0 or self.signal_period <= 0:
            raise ValueError("All periods must be positive")
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")


class MACDStrategy:
    """
    MACD-based trading strategy implementation
    
    The strategy generates:
    - BUY signal when MACD crosses above signal line
    - SELL signal when MACD crosses below signal line
    - HOLD signal otherwise
    """
    
    def __init__(self, params: Optional[MACDParameters] = None):
        """
        Initialize MACD strategy
        
        Args:
            params: MACD parameters (uses defaults if None)
        """
        self.params = params or MACDParameters()
        self.params.validate()
        
    def calculate_macd(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate MACD indicator
        
        Args:
            prices: Price series (typically close prices)
            
        Returns:
            DataFrame with MACD, signal, and histogram values
        """
        # Calculate EMAs
        ema_fast = prices.ewm(span=self.params.fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=self.params.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        signal_line = macd_line.ewm(span=self.params.signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MACD
        
        Args:
            data: DataFrame with OHLCV data (must contain 'close' column)
            
        Returns:
            Series of trading signals (1=buy, -1=sell, 0=hold)
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        # Calculate MACD
        macd_df = self.calculate_macd(data['close'])
        
        # Initialize signals with HOLD
        signals = pd.Series(Signal.HOLD.value, index=data.index)
        
        # Generate buy/sell signals based on MACD crossovers
        macd_above = macd_df['macd'] > macd_df['signal']
        macd_below = macd_df['macd'] < macd_df['signal']
        
        # Find crossover points
        buy_signals = macd_above & ~macd_above.shift(1).fillna(False)
        sell_signals = macd_below & ~macd_below.shift(1).fillna(False)
        
        signals[buy_signals] = Signal.BUY.value
        signals[sell_signals] = Signal.SELL.value
        
        return signals
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0,
                 commission: float = 0.001) -> Dict:
        """
        Backtest the MACD strategy
        
        Args:
            data: DataFrame with OHLCV data
            initial_capital: Starting capital for backtest
            commission: Transaction commission rate (0.001 = 0.1%)
            
        Returns:
            Dictionary with backtest results
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0  # Current position size
        trades = []
        equity_curve = []
        
        for i in range(len(data)):
            price = data['close'].iloc[i]
            signal = signals.iloc[i]
            
            # Record equity
            current_equity = capital + (position * price)
            equity_curve.append(current_equity)
            
            # Execute trades based on signals
            if signal == Signal.BUY.value and position == 0:
                # Buy signal - enter long position
                position_size = capital / price
                commission_cost = capital * commission
                position = position_size
                capital = 0
                
                trades.append({
                    'timestamp': data.index[i],
                    'type': 'BUY',
                    'price': price,
                    'size': position_size,
                    'commission': commission_cost
                })
                
            elif signal == Signal.SELL.value and position > 0:
                # Sell signal - close long position
                proceeds = position * price
                commission_cost = proceeds * commission
                capital = proceeds - commission_cost
                
                trades.append({
                    'timestamp': data.index[i],
                    'type': 'SELL',
                    'price': price,
                    'size': position,
                    'commission': commission_cost
                })
                
                position = 0
        
        # Close any remaining position
        if position > 0:
            final_price = data['close'].iloc[-1]
            proceeds = position * final_price
            commission_cost = proceeds * commission
            capital = proceeds - commission_cost
            position = 0
        
        # Calculate performance metrics
        final_equity = capital + (position * data['close'].iloc[-1])
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        # Calculate Sharpe ratio (assuming daily data)
        equity_series = pd.Series(equity_curve)
        daily_returns = equity_series.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 0 else 0
        
        # Calculate maximum drawdown
        equity_series = pd.Series(equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        return {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'num_trades': len(trades),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'trades': trades,
            'equity_curve': equity_curve,
            'parameters': {
                'fast_period': self.params.fast_period,
                'slow_period': self.params.slow_period,
                'signal_period': self.params.signal_period
            }
        }
    
    def optimize_parameters(self, data: pd.DataFrame, 
                          fast_range: Tuple[int, int] = (3, 30),
                          slow_range: Tuple[int, int] = (15, 70),
                          signal_range: Tuple[int, int] = (3, 20),
                          step_size: int = 5) -> Dict:
        """
        Find optimal MACD parameters through grid search
        
        Args:
            data: DataFrame with OHLCV data
            fast_range: Range for fast EMA period
            slow_range: Range for slow EMA period
            signal_range: Range for signal EMA period
            step_size: Step size for parameter search
            
        Returns:
            Dictionary with best parameters and results
        """
        best_params = None
        best_return = -float('inf')
        results = []
        tested_count = 0
        
        # Calculate total combinations
        total_combinations = 0
        for fast in range(fast_range[0], fast_range[1] + 1, step_size):
            for slow in range(slow_range[0], slow_range[1] + 1, step_size):
                for signal in range(signal_range[0], signal_range[1] + 1, step_size):
                    if fast < slow:
                        total_combinations += 1
        
        print(f"🔄 Grid Search Progress:")
        
        # Grid search over parameter space
        for fast in range(fast_range[0], fast_range[1] + 1, step_size):
            for slow in range(slow_range[0], slow_range[1] + 1, step_size):
                for signal in range(signal_range[0], signal_range[1] + 1, step_size):
                    if fast >= slow:
                        continue
                    
                    tested_count += 1
                    progress = (tested_count / total_combinations) * 100
                    
                    # Test parameters
                    params = MACDParameters(
                        fast_period=fast,
                        slow_period=slow,
                        signal_period=signal
                    )
                    
                    self.params = params
                    result = self.backtest(data)
                    
                    results.append({
                        'params': params,
                        'return': result['total_return_pct'],
                        'sharpe': result['sharpe_ratio'],
                        'max_drawdown': result['max_drawdown_pct']
                    })
                    
                    # Track best performing parameters
                    if result['total_return_pct'] > best_return:
                        best_return = result['total_return_pct']
                        best_params = params
                        print(f"   🎯 NEW BEST: Fast={fast}, Slow={slow}, Signal={signal} → Return={best_return:.2f}%")
                    
                    # Progress update
                    if tested_count % 10 == 0:
                        print(f"   📊 Progress: {tested_count}/{total_combinations} ({progress:.1f}%) - Current best: {best_return:.2f}%")
        
        return {
            'best_params': best_params,
            'best_return': best_return,
            'all_results': results
        }