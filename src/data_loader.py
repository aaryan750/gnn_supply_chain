import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta

import io

def get_sp500_tickers(limit=None):
    """Fetches the current S&P 500 tickers from Wikipedia."""
    print("Fetching S&P 500 tickers from Wikipedia...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
        table = pd.read_html(io.StringIO(html))
        df = table[0]
        # Replace dots with dashes for yfinance (e.g., BRK.B -> BRK-B)
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        if limit:
            print(f"Limiting universe to {limit} tickers for faster processing.")
            return tickers[:limit]
        return tickers
    except Exception as e:
        print("Error fetching S&P 500 from Wikipedia. Falling back to default static list.")
        # Fallback list if offline or missing lxml
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JNJ', 'JPM']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def download_data(tickers, start_date, end_date):
    """Downloads historical OHLCV data for given tickers."""
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    try:
        # Download data using yfinance
        df = yf.download(tickers, start=start_date, end=end_date, progress=True)

        # yfinance sometimes returns a MultiIndex with Price x Ticker.
        # Prefer 'Adj Close' when available, otherwise fall back to 'Close'.
        cols_level0 = []
        if hasattr(df.columns, 'levels') and len(df.columns) > 0:
            try:
                cols_level0 = list(df.columns.get_level_values(0))
            except Exception:
                cols_level0 = list(df.columns)
        else:
            cols_level0 = list(df.columns)

        if 'Adj Close' in cols_level0:
            adj_close = df['Adj Close']
        elif 'Close' in cols_level0:
            adj_close = df['Close']
        else:
            raise KeyError("Downloaded data does not contain 'Adj Close' or 'Close' columns")

        # Volume may be present under a top-level 'Volume' or as a single column.
        if 'Volume' in cols_level0:
            volume = df['Volume']
        elif 'Volume' in df.columns:
            volume = df['Volume']
        else:
            volume = None
        
        # Save raw data
        adj_close.to_csv(os.path.join(DATA_DIR, 'adj_close.csv'))
        volume.to_csv(os.path.join(DATA_DIR, 'volume.csv'))
        
        print("Data successfully downloaded and saved to data/raw/")
        return adj_close, volume
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None, None

def calculate_features(adj_close):
    """Calculates multiple rolling features for our ML nodes."""
    print("Calculating Multidimensional Node Features...")
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    # Feature 1: 5-Day Returns (Short-term momentum)
    returns_5d = adj_close.pct_change(periods=5)
    returns_5d.to_csv(os.path.join(PROCESSED_DIR, 'returns_5d.csv'))
    
    # Feature 2: 20-Day Returns (Monthly momentum)
    returns_20d = adj_close.pct_change(periods=20)
    returns_20d.to_csv(os.path.join(PROCESSED_DIR, 'returns_20d.csv'))
    
    # Feature 3: 20-Day Volatility (Risk)
    daily_returns = adj_close.pct_change(periods=1)
    volatility_20d = daily_returns.rolling(window=20).std()
    volatility_20d.to_csv(os.path.join(PROCESSED_DIR, 'volatility_20d.csv'))
    
    # Target/Correlation: 1-Day Returns
    returns_1d = daily_returns
    returns_1d.to_csv(os.path.join(PROCESSED_DIR, 'returns_1d.csv'))
    
    print("Features saved to data/processed/")
    return returns_5d, returns_20d, volatility_20d, returns_1d

if __name__ == "__main__":
    # Define time period (e.g., last 5 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # Fetch dynamically from Wikipedia (We cap at 250 to ensure API doesn't throttle too hard)
    # Removing limit=250 builds the MASSIVE model.
    tickers = get_sp500_tickers(limit=250)
    
    adj_close, volume = download_data(tickers, start_date, end_date)
    
    if adj_close is not None:
        # Calculate multidimensional features
        calculate_features(adj_close)
        
        print("Massive data pipeline executed successfully. Ready to build the massive correlation graph next.")
