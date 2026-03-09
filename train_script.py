import os
import torch
import pandas as pd
import numpy as np

from models.supply_chain_gcn import SupplyChainGCN, BaselineMLP
from src.backtester import backtest_strategy, calculate_metrics

# Set seed for reproducibility
torch.manual_seed(42)

def main():
    print("Loading processed data...")
    # Paths relative to this script
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

    try:
        returns_5d = pd.read_csv(os.path.join(DATA_PROCESSED, 'returns_5d.csv'),
                                 index_col=0, parse_dates=True)
        returns_20d = pd.read_csv(os.path.join(DATA_PROCESSED, 'returns_20d.csv'),
                                  index_col=0, parse_dates=True)
        volatility_20d = pd.read_csv(os.path.join(DATA_PROCESSED, 'volatility_20d.csv'),
                                     index_col=0, parse_dates=True)
        returns_1d = pd.read_csv(os.path.join(DATA_PROCESSED, 'returns_1d.csv'),
                                 index_col=0, parse_dates=True)
        adj_matrix = pd.read_csv(os.path.join(DATA_PROCESSED, 'adjacency_matrix.csv'),
                                 index_col=0)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure you have run 'python src/data_loader.py' and 'python src/build_graph.py' first.")
        return

    # Basic sanity checks
    if returns_5d.shape[1] != adj_matrix.shape[0] or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError('Number of tickers in returns and adjacency matrix must match')

    # Convert adjacency to edge_index
    edges = np.vstack(np.nonzero(adj_matrix.values))
    edge_index = torch.tensor(edges, dtype=torch.long)

    # Dataset dimensions
    NUM_NODES = adj_matrix.shape[0]
    NUM_FEATURES = 3  # 5-day return, 20-day return, 20-day volatility

    # Stack features to shape [T, N, Features]
    f1 = torch.tensor(returns_5d.values, dtype=torch.float)
    f2 = torch.tensor(returns_20d.values, dtype=torch.float)
    f3 = torch.tensor(volatility_20d.values, dtype=torch.float)
    
    x_all = torch.stack([f1, f2, f3], dim=-1)
    y_all = torch.tensor(returns_1d.values, dtype=torch.float)

    # Drop any time steps where features or targets contain NaNs
    valid_mask = (~torch.isnan(x_all).any(dim=-1).any(dim=-1)) & (~torch.isnan(y_all).any(dim=-1))
    x_all = x_all[valid_mask]
    y_all = y_all[valid_mask]

    print(f"Loaded data with {x_all.shape[0]} dates, {NUM_NODES} assets")
    print(f"Adjacency edges: {edge_index.shape}")
    print(f"After dropping NaNs, {x_all.shape[0]} usable time steps\n")

    # Initialize Models
    hidden_channels = 16
    gcn_model = SupplyChainGCN(num_node_features=NUM_FEATURES, hidden_channels=hidden_channels)
    mlp_model = BaselineMLP(num_node_features=NUM_FEATURES, hidden_channels=hidden_channels)

    optimizer_gcn = torch.optim.Adam(gcn_model.parameters(), lr=0.01)
    optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    def train_step(model, optimizer, x, y_true, use_graph=True):
        model.train()
        optimizer.zero_grad()
        if use_graph:
            out = model(x, edge_index)
        else:
            out = model(x)
        loss = loss_fn(out, y_true)
        loss.backward()
        optimizer.step()
        return loss.item()

    # Training Loop
    EPOCHS = 5
    print("--- Starting Training ---")
    for epoch in range(1, EPOCHS + 1):
        total_gcn_loss = 0.0
        total_mlp_loss = 0.0
        count = 0

        # We use pairs (t, t+1) where features at t predict return at t+1
        for t in range(x_all.shape[0] - 1):
            x = x_all[t].view(NUM_NODES, NUM_FEATURES)
            y_true = y_all[t + 1].view(NUM_NODES, 1)

            gcn_loss = train_step(gcn_model, optimizer_gcn, x, y_true, use_graph=True)
            mlp_loss = train_step(mlp_model, optimizer_mlp, x, y_true, use_graph=False)

            total_gcn_loss += gcn_loss
            total_mlp_loss += mlp_loss
            count += 1

        print(f"Epoch {epoch:03d} | GCN Avg Loss: {total_gcn_loss / count:.5f} | MLP Avg Loss: {total_mlp_loss / count:.5f}")

    print("\n--- Running Backtest with GCN Output ---")
    predictions = []
    actuals = []
    for t in range(x_all.shape[0] - 1):
        x = x_all[t].view(NUM_NODES, NUM_FEATURES)
        preds = gcn_model(x, edge_index).detach().numpy().flatten()
        predictions.append(preds)
        actuals.append(y_all[t + 1].numpy().flatten())

    predictions = np.stack(predictions)
    actuals = np.stack(actuals)

    returns = backtest_strategy(predictions, actuals, quantile=0.95, mode='max_return')
    calculate_metrics(returns)

if __name__ == "__main__":
    main()
