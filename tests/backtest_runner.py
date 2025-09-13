"""
Universal Backtest Runner
Supports multiple strategies: RSI, MACD, and more

Usage:
    python tests/backtest_runner.py --rsi     # Run only RSI strategy
    python tests/backtest_runner.py --macd    # Run only MACD strategy
    python tests/backtest_runner.py --all     # Run all strategies (default)
"""

import sys
import os
from pathlib import Path
import json

# Check dependencies before importing
def check_dependencies():
    """Check and report missing dependencies"""
    missing_deps = []
    install_commands = []
    
    # Check required packages
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
        'cupy': 'cupy-cuda11x',  # For GPU optimization
        'ta': 'ta',  # Technical Analysis library
        'matplotlib': 'matplotlib',  # For visualization
    }
    
    print("\n" + "="*60)
    print("📦 CHECKING DEPENDENCIES...")
    print("="*60)
    
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✅ {module_name}: Installed")
        except ImportError:
            print(f"❌ {module_name}: NOT INSTALLED")
            missing_deps.append(module_name)
            install_commands.append(pip_name)
    
    if missing_deps:
        print("\n" + "="*60)
        print("⚠️  MISSING DEPENDENCIES DETECTED!")
        print("="*60)
        print("\nThe following packages need to be installed:")
        for dep in missing_deps:
            print(f"  - {dep}")
        
        print("\n📝 INSTALLATION INSTRUCTIONS:")
        print("-" * 40)
        
        # Basic installation
        print("\n1. Basic installation (CPU only):")
        print(f"   pip install {' '.join(install_commands)}")
        
        # GPU-specific instructions
        if 'cupy' in missing_deps:
            print("\n2. For GPU acceleration (NVIDIA only):")
            print("   # Check your CUDA version first: nvidia-smi")
            print("   # For CUDA 11.x:")
            print("   pip install cupy-cuda11x")
            print("   # For CUDA 12.x:")
            print("   pip install cupy-cuda12x")
        
        if 'tensorflow' in missing_deps:
            print("\n3. TensorFlow installation options:")
            print("   # CPU only (lighter):")
            print("   pip install tensorflow-cpu")
            print("   # GPU support (requires CUDA):")
            print("   pip install tensorflow")
        
        # Complete installation command
        print("\n4. Complete installation (all features):")
        print("   pip install pandas numpy tensorflow ta matplotlib cupy-cuda11x")
        
        # Requirements file suggestion
        print("\n5. Or use requirements.txt:")
        print("   pip install -r requirements.txt")
        
        print("\n" + "="*60)
        print("After installing dependencies, run this script again.")
        print("="*60 + "\n")
        
        # Ask user if they want to continue anyway
        response = input("Do you want to continue without these packages? (y/N): ")
        if response.lower() != 'y':
            print("\nExiting. Please install the required packages and try again.")
            sys.exit(1)
        else:
            print("\n⚠️  Continuing without some features...")
            print("   Some optimizations may not work!\n")
    else:
        print("\n✅ All dependencies are installed!\n")

# Check dependencies before proceeding
check_dependencies()

# Now import the packages
try:
    import pandas as pd
except ImportError:
    print("❌ Cannot continue without pandas. Please install: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("❌ Cannot continue without numpy. Please install: pip install numpy")
    sys.exit(1)

from datetime import datetime
import random
import argparse
from typing import Dict, List, Optional
import importlib
import traceback

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.fetch_binance_data import load_binance_data
# Logging removed - only reports are generated


class UniversalBacktestRunner:
    """Universal backtest runner that can handle multiple strategies"""

    def __init__(self, data_file: str = 'btcusdt_5m.csv', gpu_only: bool = False, cpu_only: bool = False):
        self.data_file = data_file
        self.data = None
        self.results = {}
        self.sampling_info = None
        self.gpu_only = gpu_only
        self.cpu_only = cpu_only
        self.config = self._load_config()
        self.strategies = self._discover_strategies()
        # Logging removed - only reports are generated

        # Validate GPU settings
        if self.gpu_only and self.cpu_only:
            raise ValueError("Cannot set both --gpu-only and --cpu-only flags")
        
    def _load_config(self) -> Dict:
        """Load configuration from config.json"""
        config_path = Path(__file__).parent.parent / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print("⚠️  config.json not found, using defaults")
            return {}

    def _discover_strategies(self) -> Dict:
        """Discover all available strategies in the strategies folder"""
        strategies = {}
        strategies_path = Path(__file__).parent.parent / 'strategies'
        
        # Known strategy mappings
        # Use config.json for default params if available
        config_strategies = self.config.get('strategies', {}) if self.config else {}

        strategy_mappings = {
            'rsi': {
                'module': 'strategies.rsi.rsi_strategy',
                'class': 'RSIStrategy',
                'params_class': 'RSIParameters',
                'optimization_module': 'strategies.rsi.rsi_tensorflow_optimizer',
                'optimization_func': 'optimize_rsi_strategy',
                'default_params': config_strategies.get('rsi', {}).get('default_params',
                    {'period': 14, 'oversold': 30.0, 'overbought': 70.0})
            },
            'macd': {
                'module': 'strategies.macd.macd_strategy',
                'class': 'MACDStrategy',
                'params_class': 'MACDParameters',
                'optimization_module': 'strategies.macd.macd_optimization',
                'optimization_func': 'optimize_macd_strategy',
                'default_params': config_strategies.get('macd', {}).get('default_params',
                    {'fast_period': 12, 'slow_period': 26, 'signal_period': 9})
            },
            'ema': {
                'module': 'strategies.ema.ema_strategy',
                'class': 'EMAStrategy',
                'params_class': 'EMAParameters',
                'optimization_module': None,  # Will use GPU optimizer directly
                'optimization_func': None,
                'default_params': config_strategies.get('ema', {}).get('default_params',
                    {'fast_period': 12, 'slow_period': 26, 'use_signal': False})
            }
        }
        
        # Check which strategies exist
        for name, config in strategy_mappings.items():
            strategy_dir = strategies_path / name
            if strategy_dir.exists():
                strategies[name] = config
                print(f"✅ Found strategy: {name.upper()}")
        
        return strategies
        
    def load_data(self) -> pd.DataFrame:
        """Load full dataset without sampling"""
        print(f"\nLoading {self.data_file}...")
        full_data = load_binance_data(self.data_file)

        # Use full dataset
        self.data = full_data

        # Calculate years in dataset
        if len(full_data) > 0:
            time_span = full_data.index[-1] - full_data.index[0]
            years = time_span.days / 365.25

            self.sampling_info = {
                'years': years,
                'start_date': full_data.index[0],
                'end_date': full_data.index[-1],
                'size': len(full_data)
            }

            print(f"📊 Full Dataset: {years:.1f} years | {full_data.index[0].strftime('%Y-%m-%d')} to {full_data.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Total candles: {len(full_data):,}")
        else:
            print(f"⚠️  No data loaded from {self.data_file}")

        return self.data
    
    def run_strategy_backtest(self, strategy_name: str, mode: str = 'all', generate_svg: bool = True) -> Dict:
        """Run backtest for a specific strategy
        
        Args:
            strategy_name: Name of the strategy
            mode: Optimization mode ('basic-strategy', 'random-search', 'tensorflow', 'all')
        """
        if strategy_name not in self.strategies:
            print(f"❌ Strategy '{strategy_name}' not found!")
            return {}
        
        config = self.strategies[strategy_name]
        results = {}
        
        print(f"\n{'='*60}")
        print(f"🚀 RUNNING {strategy_name.upper()} STRATEGY - Mode: {mode.upper()}")
        print(f"{'='*60}")
        
        try:
            # Import strategy module
            module = importlib.import_module(config['module'])
            strategy_class = getattr(module, config['class'])
            params_class = getattr(module, config['params_class'])
            
            # 1. Simple backtest with default parameters
            if mode in ['basic-strategy', 'all']:
                print(f"\n[1/3] Simple {strategy_name.upper()}...")
                params = params_class(**config['default_params'])
                strategy = strategy_class(params)
                results['simple'] = strategy.backtest(self.data)
                self._print_summary(f'Simple {strategy_name.upper()}', results['simple'])
            
                # Logging removed - only reports generated
            
            # 2. Grid search optimization
            if mode in ['basic-strategy', 'all']:
                print(f"\n[2/3] Grid Search for {strategy_name.upper()}...")

                # Get optimization ranges from config.json
                strategy_config = self.config.get('strategies', {}).get(strategy_name, {})
                opt_ranges = strategy_config.get('optimization_ranges', {})
                grid_config = strategy_config.get('grid_search', {})

                if strategy_name == 'rsi':
                    results['grid'] = strategy.optimize_parameters(
                        self.data,
                        period_range=(opt_ranges.get('period', {}).get('min', 5),
                                    opt_ranges.get('period', {}).get('max', 30)),
                        oversold_range=(opt_ranges.get('oversold', {}).get('min', 20),
                                      opt_ranges.get('oversold', {}).get('max', 40)),
                        overbought_range=(opt_ranges.get('overbought', {}).get('min', 60),
                                        opt_ranges.get('overbought', {}).get('max', 80)),
                        step_size=grid_config.get('period_step', 5)
                    )
                elif strategy_name == 'macd':
                    results['grid'] = strategy.optimize_parameters(
                        self.data,
                        fast_range=(opt_ranges.get('fast_period', {}).get('min', 5),
                                  opt_ranges.get('fast_period', {}).get('max', 20)),
                        slow_range=(opt_ranges.get('slow_period', {}).get('min', 20),
                                  opt_ranges.get('slow_period', {}).get('max', 50)),
                        signal_range=(opt_ranges.get('signal_period', {}).get('min', 5),
                                    opt_ranges.get('signal_period', {}).get('max', 15)),
                        step_size=grid_config.get('fast_step', 5)
                    )
                elif strategy_name == 'ema':
                    results['grid'] = strategy.optimize_parameters(
                        self.data,
                        fast_range=(opt_ranges.get('fast_period', {}).get('min', 5),
                                  opt_ranges.get('fast_period', {}).get('max', 20)),
                        slow_range=(opt_ranges.get('slow_period', {}).get('min', 20),
                                  opt_ranges.get('slow_period', {}).get('max', 50)),
                        signal_range=(opt_ranges.get('signal_period', {}).get('min', 5),
                                    opt_ranges.get('signal_period', {}).get('max', 15)),
                        step_size=grid_config.get('fast_step', 5),
                        use_signal=False  # Simple EMA crossover
                    )
                print(f"✅ Best Return: {results['grid']['best_return']:.2f}%")
            
            # Log grid search results
            if 'grid' in results and 'all_results' in results['grid']:
                # Log top 5 results
                sorted_results = sorted(results['grid']['all_results'], 
                                      key=lambda x: x['return'], reverse=True)
                for res in sorted_results[:5]:
                    params = res['params']
                    # Grid search logging removed - only reports generated
            
            # Best parameters logging removed - only reports generated
            
            # 3. TensorFlow optimization (if available)
            if mode in ['tensorflow', 'all']:
                if config['optimization_module'] and config['optimization_func']:
                    print(f"\n[3/3] TensorFlow Optimization for {strategy_name.upper()}...")
                    try:
                        opt_module = importlib.import_module(config['optimization_module'])
                        optimize_func = getattr(opt_module, config['optimization_func'])
                        
                        # Run simple TensorFlow optimization
                        results['tf_simple'] = optimize_func(self.data, mode='simple')
                        if 'backtest_results' in results['tf_simple']:
                            self._print_summary(f'TF {strategy_name.upper()}', results['tf_simple']['backtest_results'])
                        
                        # Run adaptive optimization
                        print(f"\n[3b/3] TensorFlow Adaptive for {strategy_name.upper()}...")
                        results['tf_adaptive'] = optimize_func(self.data, mode='adaptive')
                        print(f"✅ Total Return: {results['tf_adaptive'].get('total_return_pct', 0):.2f}%")
                        
                    except Exception as e:
                        print(f"❌ TensorFlow error: {e}")
                        if hasattr(e, '__traceback__'):
                            traceback.print_exc()
                else:
                    print(f"\n[3/3] TensorFlow optimization not available for {strategy_name.upper()}")
            
            # 4. Random Search optimization (for RSI only)
            if mode == 'random-search' and strategy_name == 'rsi':
                print(f"\n🎲 Random Search Optimization for {strategy_name.upper()}...")
                try:
                    from strategies.rsi.rsi_random_search_optimizer import GPURSIOptimizer

                    # Initialize optimizer with GPU settings
                    gpu_only = getattr(self, 'gpu_only', False)
                    optimizer = GPURSIOptimizer(data_file=self.data_file, gpu_only=gpu_only)

                    # Use the already loaded full dataset
                    optimizer.data = self.data

                    # Get optimization ranges from config.json
                    rsi_config = self.config.get('strategies', {}).get('rsi', {}).get('optimization_ranges', {})

                    # Run optimization with 1000 random parameter combinations
                    print("\n📊 Testing 1000 parameter combinations...")
                    print(f"   Using full dataset ({self.sampling_info['years']:.1f} years)")

                    opt_results = optimizer.optimize_parameters(
                        num_combinations=1000,
                        period_range=(rsi_config.get('period', {}).get('min', 7),
                                    rsi_config.get('period', {}).get('max', 28)),
                        oversold_range=(rsi_config.get('oversold', {}).get('min', 20),
                                      rsi_config.get('oversold', {}).get('max', 40)),
                        overbought_range=(rsi_config.get('overbought', {}).get('min', 60),
                                        rsi_config.get('overbought', {}).get('max', 80))
                    )
                    
                    # SVG report generation removed
                    
                    results['random_search'] = {
                        'optimization_results': opt_results
                    }
                    
                    print(f"\n✅ Random search optimization completed!")
                    
                except ImportError as e:
                    print(f"❌ Error: Random search optimizer not available. {e}")
                    print("   Please install required dependencies: pip install cupy-cuda11x")
                except Exception as e:
                    print(f"❌ Error running random search optimization: {e}")
                    traceback.print_exc()
            
            # 5. Full GPU Vectorized optimization (RSI only for now)
            if mode == 'tensorflow-gpu' and strategy_name == 'rsi':
                print(f"\n🔥 FULL GPU VECTORIZED Optimization for {strategy_name.upper()}...")

                # Always use GPU optimizer - no fallback
                try:
                        from strategies.rsi.rsi_gpu_optimizer import GPUOptimizedRSI, GPUConfig

                        # Load ranges from config.json
                        rsi_config = self.config.get('strategies', {}).get('rsi', {}).get('optimization_ranges', {})

                        # Configure for GPU optimization
                        config = GPUConfig(
                            period_min=rsi_config.get('period', {}).get('min', 5),
                            period_max=rsi_config.get('period', {}).get('max', 50),
                            oversold_min=rsi_config.get('oversold', {}).get('min', 15.0),
                            oversold_max=rsi_config.get('oversold', {}).get('max', 35.0),
                            overbought_min=rsi_config.get('overbought', {}).get('min', 65.0),
                            overbought_max=rsi_config.get('overbought', {}).get('max', 85.0),
                            batch_size=500  # Reduced for memory efficiency
                        )

                        # Calculate total combinations
                        period_step = rsi_config.get('period', {}).get('step', 1)
                        oversold_step = rsi_config.get('oversold', {}).get('step', 1.0)
                        overbought_step = rsi_config.get('overbought', {}).get('step', 1.0)

                        period_count = int((config.period_max - config.period_min) / period_step) + 1
                        oversold_count = int((config.oversold_max - config.oversold_min) / oversold_step) + 1
                        overbought_count = int((config.overbought_max - config.overbought_min) / overbought_step) + 1
                        total_combinations = period_count * oversold_count * overbought_count

                        print(f"\n📊 GPU-Optimized Parameter Testing:")
                        print(f"   Period: {config.period_min} to {config.period_max} (step: {period_step}) = {period_count} values")
                        print(f"   Oversold: {config.oversold_min:.1f} to {config.oversold_max:.1f} (step: {oversold_step}) = {oversold_count} values")
                        print(f"   Overbought: {config.overbought_min:.1f} to {config.overbought_max:.1f} (step: {overbought_step}) = {overbought_count} values")
                        print(f"   Total combinations: {total_combinations:,}")
                        print(f"   Using full dataset: {len(self.data)} candles")

                        # Initialize GPU optimizer
                        optimizer = GPUOptimizedRSI(config=config)

                        # Run GPU-optimized parameter testing
                        opt_results = optimizer.optimize_parameters(data=self.data)

                        results['tensorflow_gpu'] = {
                            'optimization_results': opt_results,
                            'performance_stats': optimizer.performance_stats
                        }

                except ImportError as e:
                    print(f"❌ FATAL: GPU optimizer not available. {e}")
                    print("   GPU is REQUIRED for this mode!")
                    raise
                except Exception as e:
                    print(f"❌ FATAL: GPU optimization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # 5b. True GPU parallel for MACD
            elif mode == 'tensorflow-gpu' and strategy_name == 'macd':
                print(f"\n🔥 FULL GPU VECTORIZED Optimization for {strategy_name.upper()}...")
                try:
                    from strategies.macd.macd_gpu_optimizer import GPUOptimizedMACD, GPUConfig

                    # Load ranges from config.json
                    macd_config = self.config.get('strategies', {}).get('macd', {}).get('optimization_ranges', {})

                    # Configure for GPU optimization
                    config = GPUConfig(
                        fast_min=macd_config.get('fast_period', {}).get('min', 5),
                        fast_max=macd_config.get('fast_period', {}).get('max', 20),
                        fast_step=macd_config.get('fast_period', {}).get('step', 2),
                        slow_min=macd_config.get('slow_period', {}).get('min', 20),
                        slow_max=macd_config.get('slow_period', {}).get('max', 50),
                        slow_step=macd_config.get('slow_period', {}).get('step', 3),
                        signal_min=macd_config.get('signal_period', {}).get('min', 5),
                        signal_max=macd_config.get('signal_period', {}).get('max', 15),
                        signal_step=macd_config.get('signal_period', {}).get('step', 2),
                        batch_size=500
                    )

                    # Initialize GPU optimizer
                    optimizer = GPUOptimizedMACD(config=config)

                    # Run GPU-optimized parameter testing
                    opt_results = optimizer.optimize_parameters(data=self.data)

                    results['tensorflow_gpu'] = {
                        'optimization_results': opt_results,
                        'performance_stats': optimizer.performance_stats
                    }

                except Exception as e:
                    print(f"❌ FATAL: GPU optimization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

            # 5c. True GPU parallel for EMA
            elif mode == 'tensorflow-gpu' and strategy_name == 'ema':
                print(f"\n🔥 FULL GPU VECTORIZED Optimization for {strategy_name.upper()}...")
                try:
                    from strategies.ema.ema_gpu_optimizer import GPUOptimizedEMA, GPUConfig

                    # Load ranges from config.json
                    ema_config = self.config.get('strategies', {}).get('ema', {}).get('optimization_ranges', {})

                    # Configure for GPU optimization
                    config = GPUConfig(
                        fast_min=ema_config.get('fast_period', {}).get('min', 5),
                        fast_max=ema_config.get('fast_period', {}).get('max', 20),
                        fast_step=ema_config.get('fast_period', {}).get('step', 2),
                        slow_min=ema_config.get('slow_period', {}).get('min', 20),
                        slow_max=ema_config.get('slow_period', {}).get('max', 50),
                        slow_step=ema_config.get('slow_period', {}).get('step', 3),
                        batch_size=500
                    )

                    # Initialize GPU optimizer
                    optimizer = GPUOptimizedEMA(config=config)

                    # Run GPU-optimized parameter testing
                    opt_results = optimizer.optimize_parameters(data=self.data)

                    results['tensorflow_gpu'] = {
                        'optimization_results': opt_results,
                        'performance_stats': optimizer.performance_stats
                    }

                except Exception as e:
                    print(f"❌ FATAL: GPU optimization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
        except Exception as e:
            print(f"❌ Error running {strategy_name} strategy: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
        
        # Generate SVG report for RSI strategy if requested
        # Skip for tensorflow-gpu mode as it generates its own report
        if generate_svg and strategy_name == 'rsi' and results and mode != 'tensorflow-gpu':
            # SVG report generation removed
            pass
            
        return results
    
    def run_all_strategies(self, selected_strategies: Optional[List[str]] = None, mode: str = 'all'):
        """Run backtests for all or selected strategies
        
        Args:
            selected_strategies: List of strategy names to run
            mode: Optimization mode to use
        """
        strategies_to_run = selected_strategies if selected_strategies else list(self.strategies.keys())
        
        print(f"\n📋 Strategies to run: {', '.join([s.upper() for s in strategies_to_run])}")
        print(f"📋 Mode: {mode.upper()}")
        
        for strategy_name in strategies_to_run:
            strategy_results = self.run_strategy_backtest(strategy_name, mode=mode)
            if strategy_results:
                # Store results with strategy prefix
                for test_type, result in strategy_results.items():
                    self.results[f"{strategy_name}_{test_type}"] = result
    
    def _print_summary(self, name: str, result: Dict):
        """Print brief performance summary"""
        print(f"✅ {name}: Return={result.get('total_return_pct', 0):.2f}%, Sharpe={result.get('sharpe_ratio', 0):.3f}")
    
    def _generate_svg_report_for_rsi(self, results: Dict, mode: str):
        """Generate SVG report for RSI optimization results - functionality removed"""
        print("\n⚠️ SVG report generation has been removed")

    
    def _calculate_win_rate(self, result: Dict) -> float:
        """Calculate win rate from backtest result"""
        if 'trades' not in result or len(result['trades']) < 2:
            return 0
        
        trades = result['trades']
        winning_trades = 0
        total_pairs = 0
        
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i + 1]
                if buy_trade['type'] == 'BUY' and sell_trade['type'] == 'SELL':
                    if sell_trade['price'] > buy_trade['price']:
                        winning_trades += 1
                    total_pairs += 1
        
        return (winning_trades / total_pairs * 100) if total_pairs > 0 else 0
    
    def _extract_metrics(self, name: str, result: Dict) -> Dict:
        """Extract standardized metrics from different result formats"""
        if not result:
            return None
            
        # Parse strategy name from result key
        strategy_name = name.split('_')[0].upper()
        test_type = ' '.join(name.split('_')[1:]).upper()
        
        metrics = {'Strategy': f"{strategy_name} {test_type}"}
        
        if 'total_return_pct' in result:
            metrics.update({
                'Return%': result['total_return_pct'],
                'Sharpe': result.get('sharpe_ratio', 0),
                'MaxDD%': abs(result.get('max_drawdown_pct', 0)),
                'Trades': result.get('num_trades', 0)
            })
        elif 'best_return' in result:  # Grid search
            metrics.update({
                'Return%': result['best_return'],
                'Sharpe': 0,
                'MaxDD%': 0,
                'Trades': 0
            })
        elif 'backtest_results' in result:  # TensorFlow
            br = result['backtest_results']
            metrics.update({
                'Return%': br.get('total_return_pct', 0),
                'Sharpe': br.get('sharpe_ratio', 0),
                'MaxDD%': abs(br.get('max_drawdown_pct', 0)),
                'Trades': br.get('num_trades', 0)
            })
        else:
            return None
            
        return metrics
    
    def generate_report(self):
        """Generate comparison report for all strategies"""
        print("\n" + "="*60)
        print("📊 PERFORMANCE COMPARISON - ALL STRATEGIES")
        print("="*60)
        
        # Extract and compare results
        comparison = []
        for name, result in self.results.items():
            metrics = self._extract_metrics(name, result)
            if metrics:
                comparison.append(metrics)
        
        if comparison:
            # Sort by return
            comparison.sort(key=lambda x: x['Return%'], reverse=True)
            
            # Print table
            df = pd.DataFrame(comparison)
            print("\n", df.to_string(index=False))
            
            # Winner
            best = comparison[0]
            print(f"\n🏆 WINNER: {best['Strategy']} | Return: {best['Return%']:.2f}%")
            
            # Save report
            self._save_report(comparison)
    
    def _save_report(self, comparison):
        """Save report using simple logger"""
        # Logger functionality removed - only console output
        return
        
        # Save detailed parameter log
        log_filename = f'backtest_log_{timestamp}.log'
        with open(report_dir / log_filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED BACKTEST PARAMETER LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Data File: {self.data_file}\n")
            
            if self.sampling_info:
                f.write(f"\n📊 DATA SAMPLING:\n")
                f.write(f"   Period: {self.sampling_info['years']:.1f} years\n")
                f.write(f"   Start: {self.sampling_info['start_date']}\n")
                f.write(f"   End: {self.sampling_info['end_date']}\n")
                f.write(f"   Total Candles: {self.sampling_info['size']:,}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED RESULTS BY STRATEGY\n")
            f.write("="*80 + "\n")
            
            # Write detailed results for each strategy
            for name, result in self.results.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"STRATEGY: {name.upper()}\n")
                f.write(f"{'='*60}\n")
                
                if 'parameters' in result:
                    f.write("\n📈 PARAMETERS:\n")
                    params = result['parameters']
                    for key, value in params.items():
                        f.write(f"   {key}: {value}\n")
                
                if 'total_return_pct' in result:
                    f.write("\n💰 PERFORMANCE METRICS:\n")
                    f.write(f"   Total Return: {result.get('total_return_pct', 0):.2f}%\n")
                    f.write(f"   Sharpe Ratio: {result.get('sharpe_ratio', 0):.4f}\n")
                    f.write(f"   Max Drawdown: {result.get('max_drawdown_pct', 0):.2f}%\n")
                    f.write(f"   Number of Trades: {result.get('num_trades', 0)}\n")
                    f.write(f"   Initial Capital: ${result.get('initial_capital', 10000):.2f}\n")
                    f.write(f"   Final Equity: ${result.get('final_equity', 0):.2f}\n")
                
                if 'best_params' in result:  # Grid search results
                    f.write("\n🎯 OPTIMAL PARAMETERS (Grid Search):\n")
                    best_params = result['best_params']
                    if hasattr(best_params, '__dict__'):
                        for key, value in best_params.__dict__.items():
                            f.write(f"   {key}: {value}\n")
                    f.write(f"\n   Best Return: {result.get('best_return', 0):.2f}%\n")
                    
                    # Write all tested parameters
                    if 'all_results' in result:
                        f.write(f"\n📊 ALL TESTED COMBINATIONS ({len(result['all_results'])} total):\n")
                        f.write("-"*60 + "\n")
                        
                        # Sort by return
                        sorted_results = sorted(result['all_results'], 
                                              key=lambda x: x['return'], 
                                              reverse=True)
                        
                        # Show top 10 and bottom 5
                        f.write("\n🏆 TOP 10 PERFORMING PARAMETERS:\n")
                        for i, test in enumerate(sorted_results[:10], 1):
                            params = test['params']
                            f.write(f"   {i:2}. Return: {test['return']:7.2f}% | ")
                            if hasattr(params, '__dict__'):
                                param_str = ", ".join([f"{k}={v}" for k, v in params.__dict__.items()])
                                f.write(f"{param_str}\n")
                        
                        f.write("\n❌ BOTTOM 5 PERFORMING PARAMETERS:\n")
                        for i, test in enumerate(sorted_results[-5:], 1):
                            params = test['params']
                            f.write(f"   {i:2}. Return: {test['return']:7.2f}% | ")
                            if hasattr(params, '__dict__'):
                                param_str = ", ".join([f"{k}={v}" for k, v in params.__dict__.items()])
                                f.write(f"{param_str}\n")
                
                if 'optimal_parameters' in result:  # TensorFlow results
                    f.write("\n🤖 TENSORFLOW OPTIMAL PARAMETERS:\n")
                    opt_params = result['optimal_parameters']
                    for key, value in opt_params.items():
                        f.write(f"   {key}: {value}\n")
                
                if 'optimization_results' in result:  # Adaptive optimization
                    f.write(f"\n🔄 ADAPTIVE OPTIMIZATION RESULTS:\n")
                    f.write(f"   Total Windows: {len(result['optimization_results'])}\n")
                    f.write(f"   Total Return: {result.get('total_return_pct', 0):.2f}%\n")
                    f.write(f"   Final Equity: ${result.get('final_equity', 0):.2f}\n")
                    
                    # Show parameter evolution
                    f.write("\n📈 PARAMETER EVOLUTION:\n")
                    for i, window in enumerate(result['optimization_results'][:5], 1):
                        f.write(f"   Window {i}: ")
                        params = window['parameters']
                        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                        f.write(f"{param_str} | Return: {window['return']:.2f}%\n")
                
                if 'trades' in result and len(result['trades']) > 0:
                    f.write(f"\n📊 TRADE STATISTICS:\n")
                    trades = result['trades']
                    f.write(f"   Total Trades: {len(trades)}\n")
                    
                    # Calculate trade statistics
                    buy_trades = [t for t in trades if t['type'] == 'BUY']
                    sell_trades = [t for t in trades if t['type'] == 'SELL']
                    
                    f.write(f"   Buy Orders: {len(buy_trades)}\n")
                    f.write(f"   Sell Orders: {len(sell_trades)}\n")
                    
                    if buy_trades:
                        avg_buy = np.mean([t['price'] for t in buy_trades])
                        f.write(f"   Avg Buy Price: ${avg_buy:.2f}\n")
                    
                    if sell_trades:
                        avg_sell = np.mean([t['price'] for t in sell_trades])
                        f.write(f"   Avg Sell Price: ${avg_sell:.2f}\n")
                    
                    # Show first and last trades
                    f.write(f"\n   First Trade: {trades[0]['type']} @ ${trades[0]['price']:.2f} ({trades[0]['timestamp']})\n")
                    f.write(f"   Last Trade: {trades[-1]['type']} @ ${trades[-1]['price']:.2f} ({trades[-1]['timestamp']})\n")
            
            # Summary comparison at the end
            f.write("\n" + "="*80 + "\n")
            f.write("PERFORMANCE SUMMARY COMPARISON\n")
            f.write("="*80 + "\n")
            
            if comparison:
                # Header
                f.write(f"\n{'Rank':<6}{'Strategy':<30}{'Return%':<12}{'Sharpe':<10}{'MaxDD%':<10}{'Trades':<8}\n")
                f.write("-"*76 + "\n")
                
                # Data rows
                for i, result in enumerate(comparison, 1):
                    medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f"{i}."))
                    f.write(f"{medal:<6}{result['Strategy']:<30}")
                    f.write(f"{result['Return%']:>10.2f}%  ")
                    f.write(f"{result['Sharpe']:>8.4f}  ")
                    f.write(f"{result['MaxDD%']:>8.2f}%  ")
                    f.write(f"{result['Trades']:>6}\n")
                
                # Winner highlight
                if comparison:
                    best = comparison[0]
                    f.write(f"\n🏆 BEST PERFORMER: {best['Strategy']} with {best['Return%']:.2f}% return\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n📄 Detailed parameter log saved: reports/{log_filename}")
        print(f"📊 Report saved to reports/{log_filename}")


def check_gpu():
    """Check GPU availability for TensorFlow - GPU is REQUIRED"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ FATAL: No GPU found!")
            print("   GPU is REQUIRED for this optimizer!")
            print("   To enable GPU support:")
            print("   1. Install NVIDIA CUDA Toolkit")
            print("   2. Install cuDNN")
            print("   3. Reinstall TensorFlow with GPU support")
            return False
        else:
            print(f"✅ GPU Available: {gpus[0].name}")
            # Check CUDA version
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'CUDA Version' in line:
                            print(f"   CUDA Version: {line.split('CUDA Version:')[1].strip().split()[0]}")
                            break
            except:
                pass
            return True
    except ImportError:
        print("❌ FATAL: TensorFlow not installed!")
        print("   Install with: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ FATAL: Error checking GPU: {e}")
        return False


def main():
    """Main entry point with argument parsing"""
    # Show system information
    print("\n" + "="*60)
    print("🖥️  SYSTEM INFORMATION")
    print("="*60)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check if we're in the right directory
    expected_files = ['strategies', 'data', 'utils']
    missing_dirs = []
    for dir_name in expected_files:
        dir_path = Path(os.getcwd()) / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print("\n⚠️  WARNING: Some expected directories are missing:")
        for dir_name in missing_dirs:
            print(f"   - {dir_name}/")
        print("   Make sure you're running from the project root directory!")
    
    parser = argparse.ArgumentParser(
        description='Universal Backtest Runner for Trading Strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/backtest_runner.py --rsi --basic-strategy      # Run basic RSI strategy with grid search
  python tests/backtest_runner.py --rsi --random-search        # Run RSI with random search optimization
  python tests/backtest_runner.py --rsi --tensorflow           # Run RSI with TensorFlow optimization
  python tests/backtest_runner.py --macd --basic-strategy      # Run basic MACD strategy
  python tests/backtest_runner.py --macd --tensorflow          # Run MACD with TensorFlow
  python tests/backtest_runner.py --macd --tensorflow-gpu      # Run MACD with GPU optimization
  python tests/backtest_runner.py --ema --basic-strategy       # Run basic EMA crossover strategy
  python tests/backtest_runner.py --ema --tensorflow-gpu       # Run EMA with GPU optimization
        """
    )
    
    # Strategy selection arguments
    strategy_group = parser.add_argument_group('Strategy Selection')
    strategy_group.add_argument('--rsi', action='store_true', help='Run RSI strategy')
    strategy_group.add_argument('--macd', action='store_true', help='Run MACD strategy')
    strategy_group.add_argument('--ema', action='store_true', help='Run EMA crossover strategy')
    strategy_group.add_argument('--all', action='store_true', help='Run all available strategies')
    
    # Optimization mode arguments
    mode_group = parser.add_argument_group('Optimization Mode (Required)')
    mode_group.add_argument('--basic-strategy', action='store_true', 
                           help='Run basic strategy with default parameters and grid search')
    mode_group.add_argument('--random-search', action='store_true', 
                           help='Run random search optimization (RSI only)')
    mode_group.add_argument('--tensorflow', action='store_true', 
                           help='Run TensorFlow optimization')
    mode_group.add_argument('--tensorflow-gpu', action='store_true', 
                           help='Run FULL GPU VECTORIZED optimization (RSI only) - ALL tests simultaneously - 1000+ tests/sec')
    mode_group.add_argument('--all-tensorflow-gpu', action='store_true',
                           help='Run ALL strategies with TensorFlow GPU optimization sequentially')
    
    # GPU options
    gpu_group = parser.add_argument_group('GPU Options')
    gpu_group.add_argument('--gpu-only', action='store_true', 
                          help='Force GPU-only mode (fail if GPU not available)')
    gpu_group.add_argument('--cpu-only', action='store_true', 
                          help='Force CPU-only mode (disable GPU acceleration)')
    
    # Legacy support
    parser.add_argument('--gpu-rsi', action='store_true', 
                       help='(Deprecated) Use --rsi --random-search instead')
    
    args = parser.parse_args()
    
    # Handle deprecated --gpu-rsi argument
    if args.gpu_rsi:
        print("\n⚠️  --gpu-rsi is deprecated. Use --rsi --random-search instead.")
        args.rsi = True
        args.random_search = True
    
    # Check if at least one optimization mode is selected
    if not (args.basic_strategy or args.random_search or args.tensorflow or args.tensorflow_gpu or args.all_tensorflow_gpu):
        print("\n❌ Error: You must select at least one optimization mode!")
        print("   Use one of: --basic-strategy, --random-search, --tensorflow, --tensorflow-gpu, or --all-tensorflow-gpu")
        print("\nExamples:")
        print("  python tests/backtest_runner.py --rsi --basic-strategy")
        print("  python tests/backtest_runner.py --rsi --random-search")
        print("  python tests/backtest_runner.py --rsi --tensorflow")
        print("  python tests/backtest_runner.py --rsi --tensorflow-gpu  # 100%% GPU optimized!")
        print("  python tests/backtest_runner.py --all-tensorflow-gpu  # Run ALL strategies with GPU!")
        parser.print_help()
        return
    
    # Determine which strategies to run
    selected_strategies = []
    if args.rsi:
        selected_strategies.append('rsi')
    if args.macd:
        selected_strategies.append('macd')
    if args.ema:
        selected_strategies.append('ema')
    
    # Handle --all-tensorflow-gpu flag (automatically selects all GPU-enabled strategies)
    if args.all_tensorflow_gpu:
        selected_strategies = ['rsi', 'ema', 'macd']  # All GPU-enabled strategies
        mode = 'tensorflow-gpu'
        print("\n🚀 Running ALL strategies with TensorFlow GPU optimization sequentially")
        print("   Strategies: RSI, EMA, MACD")
    # If no specific strategy selected, show error
    elif not selected_strategies and not args.all:
        print("\n❌ Error: You must select at least one strategy!")
        print("   Use --rsi, --macd, --ema, --all, or --all-tensorflow-gpu")
        parser.print_help()
        return
    # If --all flag, run all available strategies
    elif args.all:
        selected_strategies = None  # This will run all available strategies
    
    # Determine optimization mode (skip if already set by --all-tensorflow-gpu)
    if not args.all_tensorflow_gpu:
        mode = 'all'
        if args.basic_strategy and not args.random_search and not args.tensorflow and not args.tensorflow_gpu:
            mode = 'basic-strategy'
        elif args.random_search and not args.basic_strategy and not args.tensorflow and not args.tensorflow_gpu:
            mode = 'random-search'
        elif args.tensorflow and not args.basic_strategy and not args.random_search and not args.tensorflow_gpu:
            mode = 'tensorflow'
        elif args.tensorflow_gpu and not args.basic_strategy and not args.random_search and not args.tensorflow:
            mode = 'tensorflow-gpu'
        elif args.basic_strategy or args.random_search or args.tensorflow or args.tensorflow_gpu:
            # Multiple modes selected - run all
            mode = 'all'
    
    # Check GPU availability
    has_gpu = check_gpu()

    # If tensorflow-gpu mode is selected, GPU is required
    if mode == 'tensorflow-gpu' and not has_gpu:
        print("\n❌ FATAL: tensorflow-gpu mode requires a GPU!")
        print("   Either:")
        print("   1. Install GPU support (CUDA + cuDNN)")
        print("   2. Use --basic-strategy mode instead")
        sys.exit(1)
    
    # Initialize and run backtest runner
    print("\n" + "="*60)
    print("🎯 UNIVERSAL BACKTEST RUNNER")
    print("="*60)
    
    # Logger is created in UniversalBacktestRunner __init__
    
    runner = UniversalBacktestRunner(gpu_only=args.gpu_only, cpu_only=args.cpu_only)
    
    # Load data
    runner.load_data()
    
    # Run selected strategies with specified mode
    runner.run_all_strategies(selected_strategies, mode=mode)
    
    # Generate comparison report
    if runner.results:
        runner.generate_report()
    else:
        print("\n❌ No results to report!")
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()