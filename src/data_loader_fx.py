import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw_fx')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed_fx')

def get_forex_universe():
    """Returns a manually curated list of 38 highly liquid Major and Minor Forex pairs available on yfinance."""
    return [
        'EURUSD=X', 'JPY=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'EURJPY=X', 'GBPJPY=X', 'EURGBP=X', 
        'EURCAD=X', 'EURSEK=X', 'EURCHF=X', 'EURHUF=X', 'EURJPY=X', 'CNY=X', 'HKD=X', 'SGD=X', 
        'INR=X', 'MXN=X', 'PHP=X', 'IDR=X', 'THB=X', 'MYR=X', 'ZAR=X', 'RUB=X', 
        'AUDCAD=X', 'AUDCHF=X', 'AUDJPY=X', 'AUDNZD=X', 'CADCHF=X', 'CADJPY=X', 'CHFJPY=X', 
        'EURAUD=X', 'EURNOK=X', 'EURTRY=X', 'GBPAUD=X', 'GBPCAD=X', 'GBPCHF=X', 'NZDCAD=X'
    ]

def download_fx_data(tickers, start_date, end_date):
    """Downloads historical daily Forex pricing from yfinance."""
    print(f"Downloading FX data for {len(tickers)} pairs from {start_date} to {end_date}...")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    try:
        df = yf.download(tickers, start=start_date, end=end_date, progress=True)

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
            raise KeyError("Downloaded data does not contain 'Close' prices")
            
        # Cleanse missing data
        threshold = int(len(adj_close) * 0.98) # Keep pairs with 98% of data
        adj_close = adj_close.dropna(axis=1, thresh=threshold)
        adj_close = adj_close.ffill().bfill()
        
        adj_close.to_csv(os.path.join(DATA_DIR, 'fx_close.csv'))
        print(f"Successfully downloaded {adj_close.shape[1]} viable FX pairs.")
        return adj_close
        
    except Exception as e:
        print(f"Error downloading FX data: {e}")
        return None

def calculate_fx_features(adj_close):
    """Calculates multiple rolling features strictly from Forex Price Momentum."""
    print("Calculating FX Multidimensional Node Features...")
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    returns_5d = adj_close.pct_change(periods=5)
    returns_5d.to_csv(os.path.join(PROCESSED_DIR, 'returns_5d.csv'))
    
    returns_20d = adj_close.pct_change(periods=20)
    returns_20d.to_csv(os.path.join(PROCESSED_DIR, 'returns_20d.csv'))
    
    daily_returns = adj_close.pct_change(periods=1)
    volatility_20d = daily_returns.rolling(window=20).std()
    volatility_20d.to_csv(os.path.join(PROCESSED_DIR, 'volatility_20d.csv'))
    
    returns_1d = daily_returns
    returns_1d.to_csv(os.path.join(PROCESSED_DIR, 'returns_1d.csv'))
    
    print("FX Features saved successfully!")

if __name__ == "__main__":
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    tickers = get_forex_universe()
    adj_close = download_fx_data(tickers, start_date, end_date)
    
    if adj_close is not None:
        calculate_fx_features(adj_close)
