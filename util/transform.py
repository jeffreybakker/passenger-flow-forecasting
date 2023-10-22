import numpy as np
import pandas as pd
import torch


class PandasToTensor(object):
    """
    Converts a Pandas DataFrame to a PyTorch Tensor.
    """

    def __call__(self, df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(df.to_numpy(dtype=np.float32))


class RollExogenousFeatures(object):
    """
    Rolls all data by one so that the historic observations contain the exogenous features for the next time step. The
    resulting tensor is one size smaller.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        res = x.clone()[:-1, :]
        res[:, 2:] = x[1:, 2:]
        return res


class GraphToTensor(object):
    def __call__(self, df: pd.DataFrame) -> torch.Tensor:
        num_nodes = df.droplevel(0).index.unique().size
        num_rows, num_features = df.shape
        return torch.tensor(df.values.reshape((num_rows // num_nodes, num_nodes, num_features)).astype(np.float32))


class GraphRollExogenousFeatures(object):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        res = x.clone()[:-1, :, :]
        res[:, :, 2:] = x[1:, :, 2:]
        return res
