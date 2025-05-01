import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

class TimeSeriesDataset(Dataset):
    def __init__(self, file_path: str, input_length: int, forecast_horizon: int):

        df = pd.read_csv('data/ETTh1.csv', parse_dates=['date'])
        vars = ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
        data = df[vars].values
        self.data = data
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.n_samples = len(data) - input_length - forecast_horizon + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_length]                     # (input_length, n_vars)
        y = self.data[idx + self.input_length : idx + self.input_length + self.forecast_horizon]
        # transpose to (n_vars, seq_len)
        return (
            torch.from_numpy(x.T).float(),
            torch.from_numpy(y.T).float()
        )

input_length    = 336   # e.g. past 336 steps
forecast_horizon= 96    # e.g. next 96 steps
batch_size      = 32

dataset = TimeSeriesDataset("data/ETTh1.csv", input_length, forecast_horizon)
n_total = len(dataset)
n_train = int(0.8 * n_total)
train_idx = list(range(0, n_train))
val_idx   = list(range(n_train, n_total))

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,     
    drop_last=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False
)


