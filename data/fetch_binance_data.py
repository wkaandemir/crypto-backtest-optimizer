"""
Binance Historical Data Fetcher - Simplified Version
Fetches all timeframes (1m to 1d) for a specified symbol
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

# ==================== CONFIGURATION ====================
# Just change this symbol to fetch data for different pairs
SYMBOL = "BTCUSDT"  # Change this to any symbol you want (SOLUSDT, ETHUSDT, etc.)

YEARS = 5           # How many years of historical data
EXCLUDE_LAST_DAYS = 1  # Exclude last N days from data

# Timeframes to fetch (1 minute to 1 day)
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]

# ========================================================


class BinanceDataFetcher:
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.klines_endpoint = "/fapi/v1/klines"
        self.max_klines_per_request = 1500
        self.request_interval = 0.5  # 500ms between requests
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_interval:
            time.sleep(self.request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, params):
        """Make API request with error handling"""
        self._rate_limit()
        
        try:
            response = requests.get(
                f"{self.base_url}{self.klines_endpoint}",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("Rate limited! Waiting 60 seconds...")
                time.sleep(60)
                return None
            else:
                print(f"API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Request error: {e}")
            return None
    
    def fetch_data(self, symbol, interval, start_time, end_time):
        """Fetch klines data for specific timeframe"""
        all_klines = []
        current_start = start_time
        max_retries = 3
        retry_count = 0
        
        print(f"Fetching {interval} data for {symbol}...")
        
        while current_start < end_time:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_time,
                'limit': self.max_klines_per_request
            }
            
            klines = self._make_request(params)
            
            if klines and len(klines) > 0:
                all_klines.extend(klines)
                last_timestamp = klines[-1][0]
                current_start = last_timestamp + 1
                retry_count = 0  # Reset retry count on success
                
                # Progress update
                progress = datetime.fromtimestamp(last_timestamp / 1000)
                print(f"  {interval}: Fetched up to {progress.strftime('%Y-%m-%d %H:%M')}")
                
                if len(klines) < self.max_klines_per_request:
                    break
            else:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed to fetch data after {max_retries} retries. Stopping.")
                    break
                print(f"Failed to fetch data, retrying... ({retry_count}/{max_retries})")
                time.sleep(5)
        
        # Convert to DataFrame
        if all_klines:
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count',
                'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            
            # Keep only OHLCV columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Clean up
            df = df.drop_duplicates(subset=['timestamp'])
            df = df.sort_values('timestamp')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        return pd.DataFrame()
    
    def save_data(self, df, symbol, interval):
        """Save DataFrame to CSV"""
        if df.empty:
            print(f"No data to save for {symbol} {interval}")
            return None
            
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Save file
        filename = f"{symbol.lower()}_{interval}.csv"
        filepath = data_dir / filename
        df.to_csv(filepath)
        
        print(f"✓ Saved {len(df)} rows to {filename}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        return filepath


def update_or_fetch_data(symbol, interval, fetcher, end_time, start_time):
    """
    Update existing data or fetch new data if file doesn't exist
    
    Returns: (DataFrame, is_update)
    """
    data_dir = Path("data")
    filename = f"{symbol.lower()}_{interval}.csv"
    filepath = data_dir / filename
    
    if filepath.exists():
        # Load existing data
        existing_df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        last_timestamp = int(existing_df.index[-1].timestamp() * 1000)
        
        # Check if we need to update
        if last_timestamp >= end_time:
            print(f"  {interval}: Already up to date")
            return existing_df, False
        
        # Fetch only new data
        print(f"  {interval}: Updating from {existing_df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        new_df = fetcher.fetch_data(symbol, interval, last_timestamp + 1, end_time)
        
        if not new_df.empty:
            # Combine old and new data
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            print(f"  Added {len(new_df)} new rows")
            return combined_df, True
        else:
            return existing_df, False
    else:
        # No existing file, fetch all data
        print(f"  {interval}: No existing file, fetching full history")
        df = fetcher.fetch_data(symbol, interval, start_time, end_time)
        return df, True


def fetch_all_timeframes(symbol=SYMBOL, years=YEARS, exclude_days=EXCLUDE_LAST_DAYS, update_only=True):
    """
    Main function: Fetches or updates all timeframes for the specified symbol
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT", "SOLUSDT")
        years: Number of years of historical data (for initial fetch)
        exclude_days: Days to exclude from the end
        update_only: If True, only fetch new data. If False, replace all data.
    """
    print("=" * 60)
    print(f"{'UPDATING' if update_only else 'FETCHING'} DATA FOR: {symbol}")
    print(f"Years: {years} | Excluding last {exclude_days} day(s)")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Mode: {'Update existing data' if update_only else 'Replace all data'}")
    print("=" * 60)
    
    # Calculate time range
    end_time = int((datetime.now() - timedelta(days=exclude_days)).timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=years*365)).timestamp() * 1000)
    
    start_date = datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d')
    end_date = datetime.fromtimestamp(end_time/1000).strftime('%Y-%m-%d')
    
    print(f"Period: {start_date} to {end_date}\n")
    
    # Initialize fetcher
    fetcher = BinanceDataFetcher()
    
    # Fetch each timeframe
    results = {}
    for interval in TIMEFRAMES:
        print(f"\n--- Processing {interval} timeframe ---")
        
        # Adjust start time for 1m data to avoid huge files
        adjusted_start = start_time
        if interval == "1m" and years > 1:
            adjusted_start = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
            print("  Note: 1m data limited to last 1 year")
        
        if update_only:
            # Update existing or fetch new
            df, needs_save = update_or_fetch_data(symbol, interval, fetcher, end_time, adjusted_start)
        else:
            # Force fetch all data
            df = fetcher.fetch_data(symbol, interval, adjusted_start, end_time)
            needs_save = True
        
        # Save data if needed
        if needs_save and not df.empty:
            filepath = fetcher.save_data(df, symbol, interval)
            results[interval] = {"rows": len(df), "file": filepath, "updated": True}
        elif not df.empty:
            results[interval] = {"rows": len(df), "file": f"data/{symbol.lower()}_{interval}.csv", "updated": False}
        else:
            results[interval] = {"rows": 0, "file": None, "updated": False}
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE - SUMMARY")
    print("=" * 60)
    for interval, info in results.items():
        if info["rows"] > 0:
            status = "✓ Updated" if info["updated"] else "○ Already up-to-date"
            print(f"{status} {interval:5s}: {info['rows']:,} total rows")
        else:
            print(f"✗ {interval:5s}: Failed")
    
    print("\nAll data in 'data/' directory")
    print("=" * 60)
    
    return results


def load_binance_data(filename):
    """Load Binance data from CSV file
    
    Args:
        filename: Name of the CSV file (e.g., 'btcusdt_5m.csv')
    
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    import pandas as pd
    from pathlib import Path
    
    # Check in data directory
    data_dir = Path(__file__).parent
    file_path = data_dir / filename
    
    if not file_path.exists():
        # Try in tests/data directory (legacy location)
        alt_path = data_dir.parent / 'tests' / 'data' / filename
        if alt_path.exists():
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Data file not found: {filename}")
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime if needed
    if 'timestamp' in df.columns:
        # Try to detect if timestamp is in milliseconds or string format
        if df['timestamp'].dtype == 'object' or df['timestamp'].dtype == 'str':
            # String format like "2020-09-02 18:50:00"
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            # Numeric format (milliseconds)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    
    # Ensure we have the required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df


if __name__ == "__main__":
    # Just change SYMBOL at the top and run!
    # Default: Updates existing data (only fetches new candles)
    fetch_all_timeframes()
    
    # To force re-download all data (replace existing):
    # fetch_all_timeframes(update_only=False)
    
    # Or call with different symbol:
    # fetch_all_timeframes(symbol="SOLUSDT")
    # fetch_all_timeframes(symbol="ETHUSDT", years=2)