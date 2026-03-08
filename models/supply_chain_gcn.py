import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class SupplyChainGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes=1):
        """
        Graph Convolutional Network for predicting cross-sectional equity returns.
        
        Args:
            num_node_features: Number of input features per node (e.g., past returns, vol)
            hidden_channels: Size of the hidden layers
            num_classes: Output size (1 for regression, projecting future return)
        """
        super(SupplyChainGCN, self).__init__()
        
        # First Graph Convolutional Layer
        # This aggregates information from immediate neighbors (e.g., direct suppliers/customers)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        
        # Second Graph Convolutional Layer
        # This aggregates information from 2-hop neighbors (e.g., supplier's suppliers)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        
        # Final linear layer to project the node embeddings to our return prediction
        self.out = nn.Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index):
        """
        Forward pass of the network.
        
        Args:
            x: Node feature matrix of shape [num_nodes, num_node_features]
            edge_index: Graph connectivity matrix of shape [2, num_edges]
        """
        # Pass features and graph structure through first GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x) # Apply non-linearity
        
        # Optional: Add dropout to prevent overfitting
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Pass through second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Final prediction layer
        out = self.out(x)
        
        return out

class BaselineMLP(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes=1):
        """
        Standard Multi-Layer Perceptron (Neural Network).
        This serves as our baseline. It DOES NOT use the supply chain graph structure.
        If our GCN beats this, we have proven 'Contagion Alpha'.
        """
        super(BaselineMLP, self).__init__()
        self.lin1 = nn.Linear(num_node_features, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.out = nn.Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index=None):
        # Note: edge_index is completely ignored here!
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        out = self.out(x)
        return out
