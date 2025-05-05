import torch
import torch.nn as nn
import torch.nn.functional as F
from PatchTST import RevIN


class PatchTSTSelfSupervised(nn.Module):
    def __init__(
        self,
        input_length,
        patch_len,
        stride,
        d_model=128,
        n_heads=8,
        n_layers=3,
        mask_ratio=0.4,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio

        self.revin = RevIN()

        # Compute number of patches
        self.n_patches = (input_length - patch_len) // stride + 1

        # Patch embedding & positional embedding
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Reconstruction head: predict original patch values
        self.reconstruction_head = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: (batch_size, nvars, input_length)
        """
        batch_size, nvars, input_length = x.shape
        x = self.revin(x, mode="norm")

        # Step 1: Create non-overlapping patches (for simplicity)
        patches = self._create_patches(x)  # (B, V, N, P)
        B, V, N, P = patches.shape

        # Step 2: Apply masking
        mask = torch.rand(B, V, N, device=x.device) < self.mask_ratio  # (B, V, N)
        masked_patches = patches.clone()
        # print(f"mask is: {mask.shape}")
        # print(f"og is: {patches.shape}")
        masked_patches = masked_patches.masked_fill(
            mask.unsqueeze(-1).expand_as(masked_patches), 0.0
        )

        # Step 3: Encode masked input
        tokens = self.patch_embed(masked_patches) + self.pos_embed  # (B, V, N, D)
        tokens = tokens.reshape(-1, N, tokens.shape[-1])  # (B*V, N, D)
        encoded = self.transformer(tokens)  # (B*V, N, D)

        # Step 4: Reconstruct patches
        reconstructed = self.reconstruction_head(encoded)  # (B*V, N, P)
        reconstructed = reconstructed.reshape(B, V, N, P)

        return reconstructed, patches, mask  # for computing loss

    def _create_patches(self, x):
        """Split x into non-overlapping patches."""
        return x.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # (B, V, N, P)


def self_supervised_loss(reconstructed, original, mask):
    # Compute loss only where mask == True
    loss = (reconstructed - original) ** 2  # (B, V, N, P)
    masked_loss = loss.masked_select(mask.unsqueeze(-1).expand_as(loss)).mean()
    return masked_loss
