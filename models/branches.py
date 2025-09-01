# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from EEGNet import EEGNet
from template_network import DTN
from stimulus import StimulusEncoder


class EEGBranch(nn.Module):
    """
    EEG branch using EEGNet as encoder.
    Input : (B,1,C,T)
    Output: (B,D_eeg) flattened EEG representation
    """
    def __init__(self, chans, samples):
        super().__init__()
        self.encoder = EEGNet(chans=chans, samples=samples)

    def forward(self, x):
        feat = self.encoder(x)                  # (B, F2, 1, T_out)
        feat = feat.view(feat.size(0), -1)      # (B, D_eeg)
        return feat


class StimulusBranch(nn.Module):
    """
    Stimulus branch using StimulusEncoder.
    Input : (B, T, 2)
    Output: (B, D_stim)
    """
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.encoder = StimulusEncoder(input_dim, hidden_dim)

    def forward(self, stim):
        feat = self.encoder(stim)  # (B, D_stim)
        return feat


class TemplateBranch(nn.Module):
    """
    Template branch using DTN.
    Input : (B, 1, C, T), labels(optional)
    Output: (B, D_temp) latent representation from DTN
    Note   : returns representation only (ignores logits)
    """
    def __init__(self, n_bands, n_features, n_channels, n_samples, n_classes, D_temp=64):
        super().__init__()
        self.network = DTN(n_bands=n_bands, n_features=n_features,
                           n_channels=n_channels, n_samples=n_samples,
                           n_classes=n_classes)

        # Projection: flatten dimension → D_temp
        self.proj = nn.Linear(32768, D_temp)

    def forward(self, x, y=None):
        logits, feat = self.network(x, y, return_feat=True) # (B, D_flat)
        feat = self.proj(feat)                              # (B, D_temp)
        return feat