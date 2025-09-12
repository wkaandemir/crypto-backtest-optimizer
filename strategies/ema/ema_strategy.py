"""
EMA (Exponential Moving Average) Crossover Trading Strategy

This module implements an EMA crossover trading strategy with configurable parameters
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
class EMAParameters:
    """EMA strategy parameters"""
    fast_period: int = 12  # Fast EMA period
    slow_period: int = 26  # Slow EMA period
    signal_period: int = 9  # Signal EMA period (optional)
    use_signal: bool = False  # Whether to use signal line
    
    def validate(self):
        """Validate parameter ranges"""
        if self.fast_period < 2:
            raise ValueError("Fast EMA period must be at least 2")
        if self.slow_period <= self.fast_period:
            raise ValueError("Slow EMA period must be greater than fast period")
        if self.use_signal and self.signal_period < 2:
            raise ValueError("Signal EMA period must be at least 2")


class EMAStrategy:
    """
    EMA crossover trading strategy implementation
    
    The strategy generates:
    - BUY signal when fast EMA crosses above slow EMA
    - SELL signal when fast EMA crosses below slow EMA
    - Optional: Use signal line for confirmation
    """
    
    def __init__(self, params: Optional[EMAParameters] = None):
        """
        Initialize EMA strategy
        
        Args:
            params: EMA parameters (uses defaults if None)
        """
        self.params = params or EMAParameters()
        self.params.validate()
        
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate EMA (Exponential Moving Average)
        
        Args:
            prices: Price series (typically close prices)
            period: EMA period
            
        Returns:
            EMA values as pandas Series
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_macd_histogram(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate MACD histogram for signal generation
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            MACD histogram values
        """
        close_prices = data['close']
        
        # Calculate EMAs
        fast_ema = self.calculate_ema(close_prices, self.params.fast_period)
        slow_ema = self.calculate_ema(close_prices, self.params.slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        if self.params.use_signal:
            # Calculate signal line (EMA of MACD)
            signal_line = macd_line.ewm(span=self.params.signal_period, adjust=False).mean()
            # Calculate histogram
            histogram = macd_line - signal_line
            return histogram
        else:
            # Use MACD line directly
            return macd_line
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on EMA crossover
        
        Args:
            data: DataFrame with OHLCV data (must contain 'close' column)
            
        Returns:
            Series of trading signals (1=buy, -1=sell, 0=hold)
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        close_prices = data['close']
        
        # Calculate EMAs
        fast_ema = self.calculate_ema(close_prices, self.params.fast_period)
        slow_ema = self.calculate_ema(close_prices, self.params.slow_period)
        
        # Initialize signals with HOLD
        signals = pd.Series(Signal.HOLD.value, index=data.index)
        
        if self.params.use_signal:
            # Use MACD histogram for signals
            histogram = self.calculate_macd_histogram(data)
            
            # Generate signals based on histogram zero crossings
            for i in range(1, len(histogram)):
                if histogram.iloc[i] > 0 and histogram.iloc[i-1] <= 0:
                    signals.iloc[i] = Signal.BUY.value
                elif histogram.iloc[i] < 0 and histogram.iloc[i-1] >= 0:
                    signals.iloc[i] = Signal.SELL.value
        else:
            # Simple EMA crossover signals
            for i in range(1, len(fast_ema)):
                # Bullish crossover (fast crosses above slow)
                if fast_ema.iloc[i] > slow_ema.iloc[i] and fast_ema.iloc[i-1] <= slow_ema.iloc[i-1]:
                    signals.iloc[i] = Signal.BUY.value
                # Bearish crossover (fast crosses below slow)
                elif fast_ema.iloc[i] < slow_ema.iloc[i] and fast_ema.iloc[i-1] >= slow_ema.iloc[i-1]:
                    signals.iloc[i] = Signal.SELL.value
        
        return signals
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0,
                 commission: float = 0.001) -> Dict:
        """
        Backtest the EMA strategy
        
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
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
        
        # Calculate maximum drawdown
        equity_series = pd.Series(equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Calculate profit factor and win rate
        profit_factor = 1.0
        win_rate = 0.0
        if len(trades) >= 2:
            gross_profits = 0
            gross_losses = 0
            winning_trades = 0
            losing_trades = 0
            
            # Pair up buy and sell trades
            for i in range(0, len(trades) - 1, 2):
                if i + 1 < len(trades):
                    buy_trade = trades[i]
                    sell_trade = trades[i + 1]
                    if buy_trade['type'] == 'BUY' and sell_trade['type'] == 'SELL':
                        profit = (sell_trade['price'] - buy_trade['price']) * buy_trade['size']
                        profit -= (buy_trade['commission'] + sell_trade['commission'])
                        
                        if profit > 0:
                            gross_profits += profit
                            winning_trades += 1
                        else:
                            gross_losses += abs(profit)
                            losing_trades += 1
            
            if gross_losses > 0:
                profit_factor = gross_profits / gross_losses
            elif gross_profits > 0:
                profit_factor = 999.99  # Cap at 999.99 when no losses
            
            total_closed_trades = winning_trades + losing_trades
            if total_closed_trades > 0:
                win_rate = (winning_trades / total_closed_trades) * 100
        
        # Calculate Calmar ratio (annualized return / max drawdown)
        calmar_ratio = 0.0
        if max_drawdown != 0:
            # Annualize the return assuming daily data
            days_in_data = len(data)
            years_in_data = days_in_data / 252  # Trading days per year
            annualized_return = total_return / years_in_data if years_in_data > 0 else total_return
            calmar_ratio = annualized_return / abs(max_drawdown)
        
        return {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'num_trades': len(trades),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'trades': trades,
            'equity_curve': equity_curve,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'parameters': {
                'fast_period': self.params.fast_period,
                'slow_period': self.params.slow_period,
                'signal_period': self.params.signal_period if self.params.use_signal else None,
                'use_signal': self.params.use_signal
            }
        }
    
    def optimize_parameters(self, data: pd.DataFrame, 
                          fast_range: Tuple[int, int] = (3, 60),
                          slow_range: Tuple[int, int] = (10, 250),
                          signal_range: Tuple[int, int] = (3, 35),
                          step_size: int = 5,
                          use_signal: bool = False) -> Dict:
        """
        Find optimal EMA parameters through grid search
        
        Args:
            data: DataFrame with OHLCV data
            fast_range: Range for fast EMA period testing
            slow_range: Range for slow EMA period testing
            signal_range: Range for signal EMA period testing
            step_size: Step size for parameter search
            use_signal: Whether to optimize signal line
            
        Returns:
            Dictionary with best parameters and results
        """
        best_params = None
        best_return = -float('inf')
        results = []
        tested_count = 0
        
        # Calculate total combinations for progress tracking
        total_combinations = 0
        for fast in range(fast_range[0], fast_range[1] + 1, step_size):
            for slow in range(slow_range[0], slow_range[1] + 1, step_size):
                if slow > fast:
                    if use_signal:
                        for signal in range(signal_range[0], signal_range[1] + 1, step_size):
                            total_combinations += 1
                    else:
                        total_combinations += 1
        
        print(f"🔄 Grid Search Progress:")
        
        # Grid search over parameter space
        for fast in range(fast_range[0], fast_range[1] + 1, step_size):
            for slow in range(slow_range[0], slow_range[1] + 1, step_size):
                if slow <= fast:
                    continue
                
                if use_signal:
                    for signal in range(signal_range[0], signal_range[1] + 1, step_size):
                        tested_count += 1
                        progress = (tested_count / total_combinations) * 100
                        
                        # Test parameters
                        params = EMAParameters(
                            fast_period=fast,
                            slow_period=slow,
                            signal_period=signal,
                            use_signal=True
                        )
                        
                        self.params = params
                        result = self.backtest(data)
                        
                        results.append({
                            'params': params,
                            'return': result['total_return_pct'],
                            'sharpe': result['sharpe_ratio'],
                            'max_drawdown': result['max_drawdown_pct'],
                            'trades': result['num_trades'],
                            'win_rate': result['win_rate'],
                            'profit_factor': result['profit_factor']
                        })
                        
                        # Track best performing parameters
                        if result['total_return_pct'] > best_return:
                            best_return = result['total_return_pct']
                            best_params = params
                            print(f"   🎯 NEW BEST: Fast={fast}, Slow={slow}, Signal={signal} → Return={best_return:.2f}%")
                        
                        # Progress update every 10 combinations
                        if tested_count % 10 == 0:
                            print(f"   📊 Progress: {tested_count}/{total_combinations} ({progress:.1f}%) - Current best: {best_return:.2f}%")
                else:
                    tested_count += 1
                    progress = (tested_count / total_combinations) * 100
                    
                    # Test parameters without signal line
                    params = EMAParameters(
                        fast_period=fast,
                        slow_period=slow,
                        use_signal=False
                    )
                    
                    self.params = params
                    result = self.backtest(data)
                    
                    results.append({
                        'params': params,
                        'return': result['total_return_pct'],
                        'sharpe': result['sharpe_ratio'],
                        'max_drawdown': result['max_drawdown_pct'],
                        'trades': result['num_trades'],
                        'win_rate': result['win_rate'],
                        'profit_factor': result['profit_factor']
                    })
                    
                    # Track best performing parameters
                    if result['total_return_pct'] > best_return:
                        best_return = result['total_return_pct']
                        best_params = params
                        print(f"   🎯 NEW BEST: Fast={fast}, Slow={slow} → Return={best_return:.2f}%")
                    
                    # Progress update every 10 combinations
                    if tested_count % 10 == 0:
                        print(f"   📊 Progress: {tested_count}/{total_combinations} ({progress:.1f}%) - Current best: {best_return:.2f}%")
        
        return {
            'best_params': best_params,
            'best_return': best_return,
            'all_results': results
        }