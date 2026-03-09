import pandas as pd
import numpy as np
import networkx as nx
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed_fx')

def build_fx_correlation_graph(returns_1d_path, threshold=0.7):
    """
    Builds an adjacency matrix based on historical forex pair correlation.
    Because FX is highly correlated via the USD, we use a much higher threshold.
    """
    print(f"Loading 1-day FX returns from {returns_1d_path}...")
    try:
        returns = pd.read_csv(returns_1d_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Could not find data. Please run data_loader_fx.py first.")
        return None, None

    # Calculate Pearson correlation matrix
    print("Calculating FX correlation matrix...")
    corr_matrix = returns.corr()

    # Create adjacency matrix based on Absolute Correlation Structure
    # If EUR/USD moves opposite to USD/JPY, they are still highly contagious.
    print(f"Applying strict Absolute threshold of |{threshold}| to create adjacency...")
    adj_matrix_bool = (corr_matrix.abs() > threshold)

    # Convert to int numpy array and remove self-loops (diagonal = 0)
    arr = adj_matrix_bool.values.astype(int).copy()
    np.fill_diagonal(arr, 0)

    # Build a DataFrame from the writable array
    adj_matrix = pd.DataFrame(arr, index=corr_matrix.index, columns=corr_matrix.columns)

    # Save the adjacency matrix
    adj_matrix.to_csv(os.path.join(PROCESSED_DIR, 'fx_adjacency_matrix.csv'))
    print("FX Adjacency matrix saved!")

    # Build NetworkX Graph for visualization/analysis
    G = nx.from_pandas_adjacency(adj_matrix)
    print(f"Built FX Currency Network Graph with {G.number_of_nodes()} pairs and {G.number_of_edges()} correlation edges.")
    
    return adj_matrix, G

if __name__ == "__main__":
    returns_file = os.path.join(PROCESSED_DIR, 'returns_1d.csv')
    
    # We use a 0.40 correlation threshold as a proxy for our "Currency Contagion" graph
    adj_matrix, G = build_fx_correlation_graph(returns_file, threshold=0.40)
