from itertools import count
from typing import Tuple

import torch
from torch import nn


class SARIMA(nn.Module):
    def __init__(self,
                 order: Tuple[int, int, int] = (0, 0, 0),
                 seasonal_lag: int = 1,
                 seasonal_order: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__()
        self._order = order
        self._seasonal_lag = seasonal_lag
        self._seasonal_order = seasonal_order

        self.sarima = nn.Linear(order[0] + order[2] + seasonal_order[0] + seasonal_order[2],
                                1,
                                True)

    def _forward(self, x: torch.Tensor):
        p = x[:, -self._order[0] - self._order[1]:, 0]
        if self._order[1] > 0:
            p = torch.diff(p, n=self._order[1])

        q = x[:, -self._order[2]:, 1] if self._order[2] > 0 else x[:, -self._order[2]:0, 1]

        P = x[:, -(self._seasonal_order[0] + self._seasonal_order[1]) * self._seasonal_lag::self._seasonal_lag, 0]
        if self._seasonal_order[1] > 0:
            P = torch.diff(P)

        Q = x[:, -self._seasonal_order[2] * self._seasonal_lag::self._seasonal_lag, 1] \
            if self._seasonal_order[2] > 0 else \
            x[:, -self._seasonal_order[2] * self._seasonal_lag:0:self._seasonal_lag, 1]

        x = torch.cat([p, q, P, Q], dim=1)
        return self.sarima(x).squeeze()

    def forward(self, history: torch.Tensor, horizon: torch.Tensor):
        max_flow = torch.max(history, dim=1).values
        max_flow[:, 1] = 1.0
        x = history.div(max_flow.unsqueeze(1))
        x = self._forward(x)
        x = torch.mul(x, max_flow[:, 0])
        return x

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

    @property
    def window_size(self) -> int:
        return max(max(self._order),
                   max(self._seasonal_order) * self._seasonal_lag)


class SARIMAX(SARIMA):
    def __init__(self,
                 order: Tuple[int, int, int] = (0, 0, 0),
                 seasonal_lag: int = 1,
                 seasonal_order: Tuple[int, int, int] = (0, 0, 0),
                 n_features: int = 0):
        super().__init__(order, seasonal_lag, seasonal_order)
        self.ex = nn.Linear(n_features, 1, bias=False) if n_features > 0 else lambda x: x

    def forward(self, history: torch.Tensor, horizon: torch.Tensor):
        max_flow = torch.max(history, dim=1).values
        max_flow[:, 1:] = 1.0
        x = history.div(max_flow.unsqueeze(1))

        a = super()._forward(x)
        b = self.ex(x[:, -1, 2:]).squeeze()

        y = torch.add(a, b)
        y = torch.mul(y, max_flow[:, 0])
        return y
