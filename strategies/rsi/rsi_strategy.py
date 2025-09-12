"""
RSI (Relative Strength Index) Trading Strategy

This module implements a RSI-based trading strategy with configurable parameters
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
class RSIParameters:
    """RSI strategy parameters"""
    period: int = 14  # RSI calculation period
    overbought: float = 70.0  # Overbought threshold (sell signal)
    oversold: float = 30.0  # Oversold threshold (buy signal)
    
    def validate(self):
        """Validate parameter ranges"""
        if self.period < 2:
            raise ValueError("RSI period must be at least 2")
        if not (0 < self.oversold < self.overbought < 100):
            raise ValueError("Invalid overbought/oversold levels")


class RSIStrategy:
    """
    RSI-based trading strategy implementation
    
    The strategy generates:
    - BUY signal when RSI crosses below oversold level
    - SELL signal when RSI crosses above overbought level
    - HOLD signal otherwise
    """
    
    def __init__(self, params: Optional[RSIParameters] = None):
        """
        Initialize RSI strategy
        
        Args:
            params: RSI parameters (uses defaults if None)
        """
        self.params = params or RSIParameters()
        self.params.validate()
        
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """
        Calculate RSI indicator
        
        Args:
            prices: Price series (typically close prices)
            
        Returns:
            RSI values as pandas Series
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=self.params.period).mean()
        avg_losses = losses.rolling(window=self.params.period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Alternative calculation for first RSI value using SMA
        avg_gains_sma = gains[:self.params.period].mean()
        avg_losses_sma = losses[:self.params.period].mean()
        
        if avg_losses_sma != 0:
            rs_sma = avg_gains_sma / avg_losses_sma
            rsi.iloc[self.params.period - 1] = 100 - (100 / (1 + rs_sma))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on RSI
        
        Args:
            data: DataFrame with OHLCV data (must contain 'close' column)
            
        Returns:
            Series of trading signals (1=buy, -1=sell, 0=hold)
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        # Calculate RSI
        rsi = self.calculate_rsi(data['close'])
        
        # Initialize signals with HOLD
        signals = pd.Series(Signal.HOLD.value, index=data.index)
        
        # Generate buy/sell signals based on RSI levels
        signals[rsi < self.params.oversold] = Signal.BUY.value
        signals[rsi > self.params.overbought] = Signal.SELL.value
        
        return signals
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0,
                 commission: float = 0.001) -> Dict:
        """
        Backtest the RSI strategy
        
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
                'period': self.params.period,
                'overbought': self.params.overbought,
                'oversold': self.params.oversold
            }
        }
    
    def optimize_parameters(self, data: pd.DataFrame, 
                          period_range: Tuple[int, int] = (3, 100),
                          oversold_range: Tuple[float, float] = (10, 50),
                          overbought_range: Tuple[float, float] = (50, 95),
                          step_size: int = 5) -> Dict:
        """
        Find optimal RSI parameters through grid search
        
        Args:
            data: DataFrame with OHLCV data
            period_range: Range for RSI period testing
            oversold_range: Range for oversold threshold
            overbought_range: Range for overbought threshold
            step_size: Step size for parameter search
            
        Returns:
            Dictionary with best parameters and results
        """
        best_params = None
        best_return = -float('inf')
        results = []
        tested_count = 0
        
        # Calculate total combinations for progress tracking
        total_combinations = 0
        for period in range(period_range[0], period_range[1] + 1, step_size):
            for oversold in np.arange(oversold_range[0], oversold_range[1] + 1, 5):
                for overbought in np.arange(overbought_range[0], overbought_range[1] + 1, 5):
                    if oversold < overbought:
                        total_combinations += 1
        
        print(f"🔄 Grid Search Progress:")
        
        # Grid search over parameter space
        for period in range(period_range[0], period_range[1] + 1, step_size):
            for oversold in np.arange(oversold_range[0], oversold_range[1] + 1, 5):
                for overbought in np.arange(overbought_range[0], overbought_range[1] + 1, 5):
                    if oversold >= overbought:
                        continue
                    
                    tested_count += 1
                    progress = (tested_count / total_combinations) * 100
                    
                    # Test parameters
                    params = RSIParameters(
                        period=period,
                        oversold=oversold,
                        overbought=overbought
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
                        print(f"   🎯 NEW BEST: Period={period}, O/S={oversold:.0f}, O/B={overbought:.0f} → Return={best_return:.2f}%")
                    
                    # Progress update every 10 combinations
                    if tested_count % 10 == 0:
                        print(f"   📊 Progress: {tested_count}/{total_combinations} ({progress:.1f}%) - Current best: {best_return:.2f}%")
        
        return {
            'best_params': best_params,
            'best_return': best_return,
            'all_results': results
        }