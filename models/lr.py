from itertools import count
from typing import Tuple

import torch
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self,
                 order: Tuple[int, int, int],
                 seasonal_lag: int,
                 seasonal_order: Tuple[int, int, int],
                 static_features: int,
                 exogenous_features: int,
                 exogenous_window: Tuple[int, int]):
        super().__init__()
        self._fc = nn.Linear(order[0] + order[2] + seasonal_order[0] + seasonal_order[2]
                             + static_features
                             + exogenous_features * (exogenous_window[1] - exogenous_window[0] + 1), 1)

        self._order = order
        self._seasonal_lag = seasonal_lag
        self._seasonal_order = seasonal_order

        self._static = static_features
        self._exogenous = exogenous_features
        self._window = exogenous_window

    def forward(self, history: torch.Tensor, horizon: torch.Tensor):
        max_flow = torch.max(history, dim=1).values
        max_flow[:, 1:] = 1.0
        history = history.div(max_flow.unsqueeze(1))

        p = history[:, -self._order[0] - self._order[1]:, 0]
        if self._order[1] > 0:
            p = torch.diff(p, n=self._order[1])

        q = history[:, -self._order[2]:, 1] if self._order[2] > 0 else history[:, -self._order[2]:0, 1]

        P = history[:, -(self._seasonal_order[0] + self._seasonal_order[1]) * self._seasonal_lag::self._seasonal_lag, 0]
        if self._seasonal_order[1] > 0:
            P = torch.diff(P)

        Q = history[:, -self._seasonal_order[2] * self._seasonal_lag::self._seasonal_lag, 1] \
            if self._seasonal_order[2] > 0 else \
            history[:, -self._seasonal_order[2] * self._seasonal_lag:0:self._seasonal_lag, 1]

        f = [
            p, q, P, Q,
            history[:, -1, -self._static - self._exogenous:-self._exogenous],
            history[:, self._window[0] - 1:, -self._exogenous:].flatten(1),
        ]
        if self._window[1] > 0:
            f.append(horizon[:, :self._window[1], -self._exogenous:].flatten(1))

        f = torch.cat(f, dim=1)
        x = self._fc(f).squeeze()
        y = torch.mul(x, max_flow[:, 0])
        return y

    def forecast(self, history: torch.Tensor, horizon: torch.Tensor, n_steps: int = 1):
        # Prepare output tensor
        res = torch.zeros((n_steps, history.shape[0]), device=history.get_device())

        # Prepare rolling window
        x = history

        # Iterate forecast horizon
        for i in count():
            # Compute forecast for step i
            res[i, :] = self.forward(x, horizon[:, i:, :])

            # Exit the loop if we have performed n steps
            if i + 1 >= n_steps:
                break

            # Otherwise, shift over the data in preparation for the next iteration
            x = x.roll(-1, 1).clone()
            x[:, -1, 0] = res[i, :]
            x[:, -1, 1:] = horizon[:, i, 1:]

        return res
