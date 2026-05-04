import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Linear residual block with BatchNorm and Dropout."""

    def __init__(self, dim, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class FCN(nn.Module):
    """
    Fully Connected Network for 2D-to-3D pose lifting.

    Based on Martinez et al., "A simple yet effective baseline for 3D human
    pose estimation" (ICCV 2017).

    Input:  (batch, 34)  — 17 joints × (x, y)
    Output: (batch, 51)  — 17 joints × (x, y, z)
    """

    def __init__(self, num_joints=17, hidden_dim=1024, num_blocks=2, dropout=0.25):
        super().__init__()
        input_dim = num_joints * 2
        output_dim = num_joints * 3

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, 34)
        x = self.input_proj(x)
        x = self.residual_blocks(x)
        return self.output_proj(x)
