from itertools import count
from typing import Tuple

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from models.encoder import FeatureEncoder
from models.hybrid import HybridMLP

import numpy as np

from models.mlp import MLP


class EventGAN(MessagePassing):
    def __init__(self, n_features: int):
        super().__init__(aggr='sum')
        self.fc = nn.Linear(n_features, n_features, bias=False)
        self.bias = nn.Parameter(torch.empty(n_features))
        # self.residual = nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # For a graph with N nodes and E edges:
        #  - x has shape [N, n_features]
        #  - edge_index has shape [2, E]
        #  - output has shape [N, n_features]

        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(1))

        # Linearly transform the node feature matrix
        x = self.fc(x)

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(1), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 1.0e-16
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate messages
        out = self.propagate(edge_index, x=x, norm=norm)

        # Apply a final bias vector
        # out += self.bias + self.residual * x
        out += self.bias

        return out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        # x_j has shape [E, n_features]

        # Normalize features
        return norm.view(-1, 1) * x_j


class STGAN(nn.Module):
    def __init__(self,
                 order: Tuple[int, int, int],
                 seasonal_lag: int,
                 seasonal_order: Tuple[int, int, int],
                 static_features: int,
                 exogenous_features: int,
                 exogenous_window: Tuple[int, int],
                 k_steps: int):
        super().__init__()
        self.spatial = EventGAN(exogenous_features)
        self.temporal = MLP(order, seasonal_lag, seasonal_order,
                                  static_features,
                                  exogenous_features, (0, 0))
        self.encoder = FeatureEncoder(exogenous_features, -exogenous_window[0], exogenous_window[1])

        self._static = static_features
        self._exogenous = exogenous_features
        self._window = exogenous_window
        self._k_steps = k_steps

    def forward(self, history: torch.Tensor, horizon: torch.Tensor, edge_index: torch.Tensor):
        # Layout of history and horizon tensors:
        # [batch, time, nodes, features]
        batch_size, T, N, F = history.size()

        # Encode features from a time window to size of `n_spatial_features`
        x = self.encoder(
            history.transpose(dim0=1, dim1=2).reshape(batch_size * N, T, F),
            horizon.transpose(dim0=1, dim1=2).reshape(batch_size * N, horizon.size(1), F))
        x = x.reshape(batch_size, N, T, F).transpose(dim0=1, dim1=2)

        # Exchange information about `n_spatial_features` exogenous features in `k_steps`
        h = x[:, -1, :, -self._exogenous:]
        for i in range(self._k_steps):
            h = self.spatial(h, edge_index)

        # Forecast flow per-node
        x = x.clone()
        x[:, -1, :, -self._exogenous:] = h
        x = x.transpose(dim0=1, dim1=2).reshape(batch_size * N, T, F)

        out = self.temporal(x, horizon)
        out = out.reshape(batch_size, N)

        return out

    def forecast(self, edge_index: torch.Tensor, history: torch.Tensor, horizon: torch.Tensor, n_steps: int = 1):
        # Prepare output tensor
        # [batch, node, time]
        res = torch.zeros((history.shape[0], history.shape[2], n_steps), device=history.get_device())

        # Prepare rolling window
        x = history

        # Iterate forecast horizon
        for i in count():
            # Compute forecast for step i
            res[:, :, i] = self.forward(x, horizon[:, i:, :, :], edge_index)

            # Exit the loop if we have performed n steps
            if i + 1 >= n_steps:
                break

            # Otherwise, shift over the data in preparation for the next iteration
            x = x.roll(-1, 1).clone()
            x[:, -1, :, 0] = res[:, :, i]
            x[:, -1, :, 1:] = horizon[:, i, :, 1:]

        return res
