import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class TimeSeriesDataset(Dataset):
    def __init__(self, file_path: str, input_length: int, forecast_horizon: int):

        df = pd.read_csv(file_path, parse_dates=["date"])
        vars = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        data = df[vars].values
        self.data = data
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.n_samples = len(data) - input_length - forecast_horizon + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_length]  # (input_length, n_vars)
        y = self.data[
            idx + self.input_length : idx + self.input_length + self.forecast_horizon
        ]
        # transpose to (n_vars, seq_len)
        return (torch.from_numpy(x.T).float(), torch.from_numpy(y.T).float())


class SelfSupervisedTimeSeriesDataset(Dataset):
    def __init__(self, file_path: str, input_length: int):
        df = pd.read_csv(file_path, parse_dates=["date"])
        vars = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        data = df[vars].values
        self.data = data
        self.input_length = input_length
        self.n_samples = len(data) - input_length + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_length]  # (input_length, n_vars)
        x = torch.from_numpy(x.T).float()  # (n_vars, input_length)
        return x
