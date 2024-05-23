import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, use_layer_norm=False):
        super().__init__()

        hidden_dim = 350

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )

        for i in range(n_layers - 2):
            if (i + 2) % 3 == 0 and use_layer_norm:
                self.mlp.append(nn.LayerNorm(hidden_dim))

            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.SiLU())

        if use_layer_norm:
            self.mlp.append(nn.LayerNorm(hidden_dim))

        self.mlp.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.mlp(x)
