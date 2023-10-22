from itertools import count

import torch
from torch import nn


class FeatureEncoder(nn.Module):
    def __init__(self, n_features: int, history: int, horizon: int):
        super().__init__()
        self._history_len = history
        self._horizon_len = horizon
        self._n_features = n_features

        size = (history + horizon) * n_features

        self._enc = nn.Sequential(
            nn.Linear(size, 2 * n_features),
            nn.ReLU(),
            nn.Linear(2 * n_features, n_features),
            nn.ReLU(),
        )

    def forward(self, history: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        x = torch.cat([history[:, -self._history_len:, -self._n_features:],
                       horizon[:, :self._horizon_len, -self._n_features:]],
                      dim=1)
        x = torch.flatten(x, start_dim=1)

        x = self._enc(x)

        # res = torch.zeros_like(history)
        # res[:, :, :] = history
        res = torch.clone(history)
        res[:, -1, -self._n_features:] = x

        return res


class FeatureEncoderModel(nn.Module):
    def __init__(self, model: nn.Module, encoder: nn.Module):
        super().__init__()
        self.model = model
        self.encoder = encoder

    def forward(self, history, horizon):
        x = self.encoder(history, horizon)
        return self.model(x)

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
