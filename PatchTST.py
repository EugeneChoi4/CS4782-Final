import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head
        self.head = nn.Linear(d_model * self.n_patches, forecast_horizon)

    def forward(self, x):
        """
        x: (batch_size, input_length)
        """
        x = self.revin(x, mode='norm')
        patches = self._create_patches(x)  # (B, N, P)
        x = self.patch_embed(patches) + self.pos_embed  # (B, N, D)
        z = self.transformer(x)  # (B, N, D)
        z = z.reshape(z.size(0), -1)  # (B, N*D)
        out = self.head(z)  # (B, T)
        out = self.revin(out, mode='denorm')
        return out

    def _create_patches(self, x):
        """Split x into overlapping patches."""
        B, L = x.shape
        patches = []
        for i in range(0, L - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len]  # (B, P)
            patches.append(patch.unsqueeze(1))
        return torch.cat(patches, dim=1)  # (B, N, P)


# Example usage
if __name__ == '__main__':
    B, L, T = 32, 336, 96  # Batch size, input length, forecast horizon
    x = torch.randn(B, L)
    model = PatchTST(input_length=L, patch_len=16, stride=8, forecast_horizon=T)
    y_pred = model(x)
    print(y_pred.shape)  # Expected: (32, 96)
