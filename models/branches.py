# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from EEGNet import EEGNet
from template_network import DTN
from stimulus import StimulusEncoder


class EEGBranch(nn.Module):
    """
    EEG branch using EEGNet as encoder.
    Input : (B, 1, C, T)
    Output : (B, D_eeg) flattened EEG representation
    """
    def __init__(self, chans, samples):
        super().__init__()
        self.encoder = EEGNet(chans=chans, samples=samples)
        self.out_dim = self.encoder.out_dim  # 자동 계산된 dimension

    def forward(self, x):
        feat = self.encoder(x)  # (B, out_dim)
        return feat


class StimulusBranch(nn.Module):
    """
    Stimulus branch using StimulusEncoder.
    Input : (B, T, 2) sinusoidal references (sin, cos)
    Output: (B, D_stim)
    """
    def __init__(self, hidden_dim=128, n_harmonics=3):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.encoder = StimulusEncoder(in_dim=2 * n_harmonics, hidden_dim=hidden_dim)

    def forward(self, stim):
        """
        stim: (B, T, 2) fundamental sin/cos
        Returns: (B, D_stim)
        """
        base_sin, base_cos = stim[..., 0], stim[..., 1]              # (B, T)

        harmonics = []
        for h in range(1, self.n_harmonics + 1):
            harmonics.append(torch.sin(h * torch.arcsin(base_sin)))  # sin(hf)
            harmonics.append(torch.cos(h * torch.arccos(base_cos)))  # cos(hf)

        stim_harm = torch.stack(harmonics, dim=-1)  # (B, T, 2*n_harmonics)
        feat = self.encoder(stim_harm)              # (B, hidden_dim) = (B, D_stim)
        return feat


class TemplateBranch(nn.Module):
    """
    Template branch using DTN.
    Input : (B, 1, C, T), labels (optional)
    Output : (B, D_temp) latent representation
    """
    def __init__(self, n_bands, n_features, n_channels, n_samples, n_classes, D_temp=64):
        super().__init__()
        self.network = DTN(n_bands=n_bands, n_features=n_features,
                           n_channels=n_channels, n_samples=n_samples,
                           n_classes=n_classes)

        # Projection: (B, n_features) → (B, D_temp)
        self.proj = nn.Linear(n_features, D_temp)

    def forward(self, x, y=None):
        _, feat = self.network(x, y, return_feat=True)      # (B, n_features)
        feat = self.proj(feat)                              # (B, D_temp)
        return feat