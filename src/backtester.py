import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

PROCESSED_DIR = '../data/processed'

def backtest_strategy(predictions, actual_returns, quantile=0.9, mode='long_short'):
    """
    Vectorized backtester.
    mode='long_short': Longs top quantile, Shorts bottom quantile (Market Neutral, Low Risk).
    mode='max_return': Heavily concentrates 100% capital into the Top 5 highest predicted assets (High Risk, Max Return).
    """
    if mode == 'max_return':
        print(f"Running backtest in MAX RETURN mode: 100% Long on Top {1-quantile:.0%} assets.")
        
        long_signals = (predictions >= np.quantile(predictions, quantile, axis=1, keepdims=True)).astype(int)
        long_weights = long_signals / np.sum(long_signals, axis=1, keepdims=True)
        long_weights = np.nan_to_num(long_weights)
        
        strategy_returns = np.sum(long_weights * actual_returns, axis=1)
        return strategy_returns

    else:
        print(f"Running backtest... Long Top {1-quantile:.0%}, Short Bottom {1-quantile:.0%}")
        
        # Create signals based on predictions
        long_signals = (predictions >= np.quantile(predictions, quantile, axis=1, keepdims=True)).astype(int)
        short_signals = (predictions <= np.quantile(predictions, 1 - quantile, axis=1, keepdims=True)).astype(int)
        
        # Weights (equal weight among selected)
        long_weights = long_signals / np.sum(long_signals, axis=1, keepdims=True)
        short_weights = short_signals / np.sum(short_signals, axis=1, keepdims=True)
        
        # Handle NaNs if no assets selected
        long_weights = np.nan_to_num(long_weights)
        short_weights = np.nan_to_num(short_weights)
        
        # Portfolio Weights (Long - Short)
        portfolio_weights = long_weights - short_weights
        
        # Calculate Strategy Returns
        strategy_returns = np.sum(portfolio_weights * actual_returns, axis=1)
        
        return strategy_returns

def calculate_metrics(strategy_returns):
    """Calculates standard quantitative finance metrics."""
    # Assuming daily returns
    ann_factor = 252
    
    total_return = np.prod(1 + strategy_returns) - 1
    ann_return = (1 + total_return) ** (ann_factor / len(strategy_returns)) - 1
    
    ann_vol = np.std(strategy_returns) * np.sqrt(ann_factor)
    
    sharpe_ratio = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Calculate Max Drawdown
    cum_returns = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    print("\n--- Strategy Performance ---")
    print(f"Annualized Return: {ann_return:.2%}")
    print(f"Annualized Volatility: {ann_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    return {
        'Return': ann_return,
        'Vol': ann_vol,
        'Sharpe': sharpe_ratio,
        'MaxDD': max_drawdown
    }

if __name__ == "__main__":
    print("Backtester engine ready for model outputs.")
