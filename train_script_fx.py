import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from models.supply_chain_gcn import SupplyChainGCN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed_fx')

# Hyperparameters for FX
NUM_FEATURES = 3 
HIDDEN_CHANNELS = 64
EPOCHS = 50
LEARNING_RATE = 0.005

def load_fx_data():
    """Loads features and adjacency matrix into PyTorch format."""
    print("Loading FX processed data...")
    
    returns_1d = pd.read_csv(os.path.join(PROCESSED_DIR, 'returns_1d.csv'), index_col=0, parse_dates=True)
    returns_5d = pd.read_csv(os.path.join(PROCESSED_DIR, 'returns_5d.csv'), index_col=0, parse_dates=True)
    returns_20d = pd.read_csv(os.path.join(PROCESSED_DIR, 'returns_20d.csv'), index_col=0, parse_dates=True)
    volatility_20d = pd.read_csv(os.path.join(PROCESSED_DIR, 'volatility_20d.csv'), index_col=0, parse_dates=True)
    
    adj_matrix = pd.read_csv(os.path.join(PROCESSED_DIR, 'fx_adjacency_matrix.csv'), index_col=0)
    
    # Align dates
    common_dates = returns_1d.index.intersection(returns_5d.index).intersection(returns_20d.index).intersection(volatility_20d.index)
    
    returns_1d = returns_1d.loc[common_dates]
    returns_5d = returns_5d.loc[common_dates]
    returns_20d = returns_20d.loc[common_dates]
    volatility_20d = volatility_20d.loc[common_dates]
    
    # Build Edges
    edge_indices = np.where(adj_matrix.values == 1)
    edge_index = torch.tensor(list(zip(edge_indices[0], edge_indices[1])), dtype=torch.long).t().contiguous()
    
    # Pre-compute Target (Next day's return)
    # Target[t] = returns_1d[t+1]
    target_returns = returns_1d.shift(-1)
    
    num_nodes = len(adj_matrix.columns)
    print(f"Loaded FX data with {len(common_dates)} dates, {num_nodes} currency pairs")
    print(f"FX Adjacency edges: {edge_index.shape}")
    
    # Store everything as tensors per timestep
    data_list = []
    
    for i, date in enumerate(common_dates[:-1]): # Drop last row since we don't have tomorrow's target
        # Features: Shape [NUM_NODES, NUM_FEATURES]
        f1 = torch.tensor(returns_5d.iloc[i].values, dtype=torch.float)
        f2 = torch.tensor(returns_20d.iloc[i].values, dtype=torch.float)
        f3 = torch.tensor(volatility_20d.iloc[i].values, dtype=torch.float)
        
        # Combine explicitly
        x = torch.stack([f1, f2, f3], dim=1) 
        
        y = torch.tensor(target_returns.iloc[i].values, dtype=torch.float)
        
        # We only pass data where all features for a node are valid
        valid_mask = ~torch.isnan(x).any(dim=1) & ~torch.isnan(y)
        
        if valid_mask.sum() > 0:
            data = Data(x=x, edge_index=edge_index, y=y, valid_mask=valid_mask, date=date)
            data_list.append(data)
            
    print(f"After dropping NaNs, {len(data_list)} usable time steps")
    return data_list, num_nodes

def simulate_max_conviction_backtest(predictions, actuals, mask):
    """
    To achieve near-zero drawdown, we only trade the #1 Absolute Highest 
    Conviction Long and #1 Absolute Highest Conviction Short pair each day.
    """
    portfolio_returns = []
    
    for t in range(len(predictions)):
        pred_t = predictions[t]
        actual_t = actuals[t]
        mask_t = mask[t]
        
        # Ensure mask is 1D boolean array
        mask_t = mask_t.squeeze()
        valid_indices = torch.where(mask_t)[0]
        
        if len(valid_indices) < 2:
            portfolio_returns.append(0.0)
            continue
            
        valid_preds = pred_t[mask_t]
        valid_actuals = actual_t[mask_t]
        
        # If the network predicts the exact same value for multiple FX pairs (due to lack of variance), 
        # add microscopic noise so argmax and argmin don't select the exact same pair.
        noise = torch.randn_like(valid_preds) * 1e-7
        noisy_preds = valid_preds + noise
        
        # Find exactly 1 Long and 1 Short relative to the *valid* array
        # argmax returns the index [0, len(valid_preds)-1]
        local_long_idx = torch.argmax(noisy_preds).item()
        local_short_idx = torch.argmin(noisy_preds).item()
        
        # Equal Weight allocation (50% Long, -50% Short)
        long_ret = valid_actuals[local_long_idx].item()
        short_ret = -valid_actuals[local_short_idx].item()
        
        daily_ret = (long_ret + short_ret) / 2.0
        portfolio_returns.append(daily_ret)
        
    return portfolio_returns

def run_project():
    data_list, num_nodes = load_fx_data()
    
    # 70/30 Train/Test Split
    train_size = int(len(data_list) * 0.7)
    train_data = data_list[:train_size]
    test_data = data_list[train_size:]
    
    model_gcn = SupplyChainGCN(num_node_features=NUM_FEATURES, hidden_channels=HIDDEN_CHANNELS)
    optimizer_gcn = torch.optim.Adam(model_gcn.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    
    # Training Loop
    print("\n--- Starting FX GCN Training ---")
    model_gcn.train()
    for epoch in range(1, EPOCHS + 1):
        gcn_loss_sum = 0
        
        for data in train_data:
            mask = data.valid_mask
            if mask.sum() == 0: continue
            
            # GCN
            optimizer_gcn.zero_grad()
            out_gcn = model_gcn(data.x, data.edge_index).squeeze()
            loss_gcn = criterion(out_gcn[mask], data.y[mask])
            loss_gcn.backward()
            optimizer_gcn.step()
            gcn_loss_sum += loss_gcn.item()
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | GCN Avg Loss: {gcn_loss_sum/len(train_data):.5f}")
            
    # Evaluation Loop (Out-of-sample)
    print("\n--- Running Max Conviction Backtest ---")
    model_gcn.eval()
    
    all_preds = []
    all_actuals = []
    all_masks = []
    
    with torch.no_grad():
        for data in test_data:
            out_gcn = model_gcn(data.x, data.edge_index).squeeze()
            all_preds.append(out_gcn)
            all_actuals.append(data.y)
            all_masks.append(data.valid_mask)
            
    # Run the Near-Zero Drawdown Backtester
    daily_returns = simulate_max_conviction_backtest(all_preds, all_actuals, all_masks)
    
    # Calculate Institutional Metrics
    ret_series = pd.Series(daily_returns)
    ret_series.to_csv("fx_backtest_daily.csv")
    
    # Multiply by 252 (Trading days in a year for Forex logic)
    ann_return = ret_series.mean() * 252
    ann_vol = ret_series.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    cum_returns = (1 + ret_series).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    print("\n--- Forex Strategy Performance ---")
    print(f"Annualized Return: {ann_return*100:.2f}%")
    print(f"Annualized Volatility: {ann_vol*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")

if __name__ == "__main__":
    run_project()
