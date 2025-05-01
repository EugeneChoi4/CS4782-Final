import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = x.mean(dim=-1, keepdim=True)
            self.std = x.std(dim=-1, keepdim=True) + self.eps
            x = (x - self.mean) / self.std
        elif mode == 'denorm':
            x = x * self.std + self.mean
        return x


class PatchTST(nn.Module):
    def __init__(self, input_length, patch_len, stride, forecast_horizon, d_model=128, n_heads=8, n_layers=3):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.forecast_horizon = forecast_horizon

        self.revin = RevIN()

        # Number of patches
        self.n_patches = (input_length - patch_len) // stride + 1

        # Patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # Prediction head
        self.head = nn.Linear(d_model * self.n_patches, forecast_horizon)

    def forward(self, x):
        """
        x: (batch_size, nvars, input_length)
        """
        batch_size, nvars, input_length = x.shape

        x = self.revin(x, mode='norm')
        patches = self._create_patches(x)
        print("Patches shape:", patches.shape)
        # x: (batch_size, nvars, patch_num, patch_len)
        x = self.patch_embed(patches) + self.pos_embed
        print("After embedding shape:", x.shape)
        # x: (batch_size, nvars, patch_num, d_model)
        # use the encoder for each variable independently
        x = x.permute(0, 2, 1, 3).reshape(-1, self.n_patches, x.shape[-1])
        x = self.transformer(x)
        # x: (batch_size * nvars, patch_num, d_model)
        x = x.reshape(-1, self.n_patches * x.shape[-1])
        x = self.head(x)
        print("After prediction head shape:", x.shape)
        # x: (batch_size * nvars, forecast_horizon)
        x = x.reshape(batch_size, nvars, self.forecast_horizon)
        x = self.revin(x, mode='denorm')
        return x

    def _create_patches(self, x):
        """Split x into overlapping patches."""
        batch_size, nvars, input_length = x.shape
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x


# Example usage
if __name__ == '__main__':
    # Batch size, input length, forecast horizon
    batch_size, nvars, input_length, target_window = 32, 8, 336, 96
    x = torch.randn(batch_size, nvars, input_length)  # Example input
    model = PatchTST(input_length=input_length, patch_len=16,
                     stride=8, forecast_horizon=target_window)
    y_pred = model(x)
    print(y_pred.shape)  # Expected: (32, 8, 96)
