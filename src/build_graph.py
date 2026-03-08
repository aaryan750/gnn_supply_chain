import pandas as pd
import numpy as np
import networkx as nx
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def build_correlation_graph(returns_1d_path, threshold=0.5):
    """
    Builds an adjacency matrix based on historical price correlation.
    If the correlation between two stocks is > threshold, they have an edge.
    """
    print(f"Loading 1-day returns from {returns_1d_path}...")
    try:
        returns = pd.read_csv(returns_1d_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Could not find {returns_1d_path}. Please run data_loader.py first.")
        return None, None

    # Calculate Pearson correlation matrix
    print("Calculating correlation matrix...")
    corr_matrix = returns.corr()

    # Create adjacency matrix
    print(f"Applying threshold of {threshold} to create adjacency matrix...")
    adj_matrix_bool = (corr_matrix > threshold)

    # Convert to int numpy array and remove self-loops (diagonal = 0)
    arr = adj_matrix_bool.values.astype(int).copy()
    np.fill_diagonal(arr, 0)

    # Build a DataFrame from the writable array
    adj_matrix = pd.DataFrame(arr, index=corr_matrix.index, columns=corr_matrix.columns)

    # Save the adjacency matrix
    adj_matrix.to_csv(os.path.join(PROCESSED_DIR, 'adjacency_matrix.csv'))
    print("Adjacency matrix saved to data/processed/adjacency_matrix.csv")

    # Build NetworkX Graph for visualization/analysis
    G = nx.from_pandas_adjacency(adj_matrix)
    print(f"Built Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    return adj_matrix, G

if __name__ == "__main__":
    returns_file = os.path.join(PROCESSED_DIR, 'returns_1d.csv')
    
    # We use a 0.2 correlation threshold as a proxy for our "Supply Chain/Contagion" graph
    adj_matrix, G = build_correlation_graph(returns_file, threshold=0.2)
    
    if adj_matrix is not None:
        print("Graph construction complete. Ready for PyTorch Geometric model building.")
