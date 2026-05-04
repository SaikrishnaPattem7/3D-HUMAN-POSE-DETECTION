import torch
import torch.nn as nn
import numpy as np

from data.preprocess import build_adjacency_matrix


class GraphConvBlock(nn.Module):
    """
    Single graph convolution layer: A @ X @ W
    followed by BatchNorm, ReLU, and Dropout.
    """

    def __init__(self, in_features, out_features, A, dropout=0.25):
        super().__init__()
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, J, F)
        x = torch.matmul(self.A, x)           # graph aggregation: (B, J, F)
        B, J, F = x.shape
        x = self.fc(x.reshape(B * J, F))      # linear projection: (B*J, out)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x.reshape(B, J, -1)            # (B, J, out)


class GCN(nn.Module):
    """
    Graph Convolutional Network for 2D-to-3D pose lifting.

    Models the human skeleton as a graph where joints are nodes and
    bones are edges. The adjacency matrix encodes COCO skeleton connectivity.

    Input:  (batch, 34) flat  or  (batch, 17, 2)
    Output: (batch, 51) flat  — 17 joints × (x, y, z)
    """

    def __init__(self, num_joints=17, hidden_dim=128, num_layers=4, dropout=0.25):
        super().__init__()
        A = build_adjacency_matrix(num_joints)

        layers = [GraphConvBlock(2, hidden_dim, A, dropout)]
        for _ in range(num_layers - 1):
            layers.append(GraphConvBlock(hidden_dim, hidden_dim, A, dropout))
        self.graph_convs = nn.ModuleList(layers)

        self.output = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        # Accept flat (B, 34) or shaped (B, 17, 2)
        if x.dim() == 2:
            x = x.reshape(x.shape[0], -1, 2)   # (B, 17, 2)

        for layer in self.graph_convs:
            x = layer(x)                         # (B, 17, hidden_dim)

        x = self.output(x)                       # (B, 17, 3)
        return x.reshape(x.shape[0], -1)         # (B, 51)
