# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class StimulusEncoder(nn.Module):
    """
    Stimulus Encoder
    - Encodes sinusoidal reference signals (sin, cos) into latent features.
    - Specifically designed for SSVEP tasks where frequency/phase patterns are important.
    - Input: (B, T, 2) -> B=batch, T=time length, 2=[sin, cos]
    - Output: (B, hidden_dim) stimulus embedding
    """
    def __init__(self, in_dim=2, hidden_dim=64):
        """
        Args:
            in_dim (int): Input dimension, usually 2 (sin, cos)
            hidden_dim (int): Output embedding dimension
        """
        super().__init__()

        # Conv-based encoder: preserves temporal structure and frequency information
        self.encoder = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # global temporal pooling → single embedding
        )

    def forward(self, stim):
        """
        Args:
            stim (Tensor): (B, T, 2) sinusoidal stimulus signals

        Returns:
            Tensor: (B, hidden_dim) stimulus embedding
        """
        stim = stim.permute(0, 2, 1)    # (B, T, 2) -> (B, 2, T)
        feat = self.encoder(stim)       # (B, hidden_dim, 1)
        feat = feat.squeeze(-1)         # (B, hidden_dim)

        return feat
