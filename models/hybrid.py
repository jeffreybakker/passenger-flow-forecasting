from typing import Tuple

import torch
from torch import nn

from models.sarima import SARIMA


class HybridLR(SARIMA):
    def __init__(self,
                 order: Tuple[int, int, int] = (0, 0, 0),
                 seasonal_lag: int = 1,
                 seasonal_order: Tuple[int, int, int] = (0, 0, 0),
                 static_features: int = 0,
                 exogenous_features: int = 0,
                 exogenous_window: Tuple[int, int] = (0, 0)):
        super().__init__(order, seasonal_lag, seasonal_order)
        self._fc = nn.Linear(static_features
                             + exogenous_features * (exogenous_window[1] - exogenous_window[0] + 1)
                             + 1, 1)

        self._static = static_features
        self._exogenous = exogenous_features
        self._window = exogenous_window

    def forward(self, history: torch.Tensor, horizon: torch.Tensor):
        max_flow = torch.max(history, dim=1).values
        max_flow[:, 1:] = 1.0
        history = history.div(max_flow.unsqueeze(1))

        sarima = super()._forward(history)

        f = [
            sarima.unsqueeze(1),
            history[:, -1, -self._static-self._exogenous:-self._exogenous],
            history[:, self._window[0]-1:, -self._exogenous:].flatten(1),
        ]
        if self._window[1] > 0:
            f.append(horizon[:, :self._window[1], -self._exogenous:].flatten(1))

        f = torch.cat(f, dim=1)
        x = self._fc(f).squeeze()

        y = torch.add(sarima, x)
        y = torch.mul(y, max_flow[:, 0])
        return y


class HybridMLP(SARIMA):
    def __init__(self,
                 order: Tuple[int, int, int] = (0, 0, 0),
                 seasonal_lag: int = 1,
                 seasonal_order: Tuple[int, int, int] = (0, 0, 0),
                 static_features: int = 0,
                 exogenous_features: int = 0,
                 exogenous_window: Tuple[int, int] = (0, 0)):
        super().__init__(order, seasonal_lag, seasonal_order)
        self._l1 = nn.Linear(static_features
                             + exogenous_features * (exogenous_window[1] - exogenous_window[0] + 1)
                             + 1, 16)
        self._l2 = nn.Linear(16, 8)
        self._l3 = nn.Linear(8, 1)

        self._static = static_features
        self._exogenous = exogenous_features
        self._window = exogenous_window

    def forward(self, history: torch.Tensor, horizon: torch.Tensor):
        max_flow = torch.max(history, dim=1).values
        max_flow[:, 1:] = 1.0
        history = history.div(max_flow.unsqueeze(1))

        sarima = super()._forward(history)

        f = [
            sarima.unsqueeze(1),
            history[:, -1, -self._static - self._exogenous:-self._exogenous],
            history[:, self._window[0] - 1:, -self._exogenous:].flatten(1),
        ]
        if self._window[1] > 0:
            f.append(horizon[:, :self._window[1], -self._exogenous:].flatten(1))

        f = torch.cat(f, dim=1)

        x = torch.relu(self._l1(f)).squeeze()
        x = torch.relu(self._l2(x))
        x = self._l3(x).squeeze()

        y = torch.add(sarima, x)
        y = torch.mul(y, max_flow[:, 0])
        return y
