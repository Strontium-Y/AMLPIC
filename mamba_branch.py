import torch
import torch.nn as nn

class MambaBranch(nn.Module):
    """
    Lightweight Mamba branch for extracting spectral-spatial sequence features.
    Input: (B, 1, C, H, W) where C=200, H=W=13
    Output: (B, dim) feature vector
    """
    def __init__(self, spec_dim=200, patch_size=13, dim=128, d_state=64, d_conv=4, expand_factor=2):
        super().__init__()
        self.spec_dim = spec_dim
        self.patch_size = patch_size
        self.dim = dim

        # Attempt to import Mamba
        try:
            from mamba_ssm import Mamba
            self.use_mamba = True
        except ImportError:
            print("Mamba not installed, using simplified SSM (S4) fallback")
            self.use_mamba = False
            self._init_fallback()

        if self.use_mamba:
            # Project spectral vector of each pixel to dim
            self.proj = nn.Linear(spec_dim, dim)
            # Mamba layer
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
            )
            self.norm = nn.LayerNorm(dim)

    def _init_fallback(self):
        """Simplified LSTM fallback when Mamba is not available"""
        self.proj = nn.Linear(self.spec_dim, self.dim)
        self.lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, x):
        # x: (B, 1, C, H, W)
        B, _, C, H, W = x.shape
        # Flatten spatial dimensions: (B, C, H*W)
        x = x.view(B, C, H * W)
        # Transpose to (B, L, C) where L = H*W
        x = x.permute(0, 2, 1)  # (B, L, C)
        # Project to dim
        x = self.proj(x)  # (B, L, dim)

        if self.use_mamba:
            x = self.mamba(x)  # (B, L, dim)
            x = self.norm(x)
        else:
            x, _ = self.lstm(x)
            x = self.norm(x)

        # Global average pooling: (B, dim)
        x = x.mean(dim=1)
        return x