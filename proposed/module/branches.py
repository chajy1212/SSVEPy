# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

from eegnet import EEGNet
from dtn import DTN
from stimulus import StimulusEncoder


class EEGBranch(nn.Module):
    """
    EEG branch using EEGNet as encoder.
    Input : (B, 1, C, T)
    Output:
      - return_sequence=False: (B, D_flat = C*W)
      - return_sequence=True : (B, N=W, D=C)  ← DualAttention
    """
    def __init__(self, chans, samples):
        super().__init__()
        self.encoder = EEGNet(chans=chans, samples=samples)
        self.out_dim = self.encoder.out_dim
        self.feature_dim = self.encoder.feature_dim
        self.sequence_len = self.encoder.sequence_len

    def forward(self, x, return_sequence=False):
        return self.encoder(x, return_sequence=return_sequence)


class StimulusBranch(nn.Module):
    """
    Stimulus branch using StimulusEncoder.
    Input : (B, T, 2) sinusoidal references (sin, cos)
    Output: (B, D_stim)
    """
    def __init__(self, T, sfreq=250.0, hidden_dim=128, n_harmonics=3):
        super().__init__()
        self.T = T
        self.sfreq = sfreq
        self.n_harmonics = n_harmonics
        self.encoder = StimulusEncoder(in_dim=2 * n_harmonics, hidden_dim=hidden_dim)

        # Precompute time vector for efficiency
        t = torch.arange(self.T, dtype=torch.float32) / self.sfreq
        self.register_buffer("t", t)

    def forward(self, freqs):
        """
        Args:
            freq: (B,) corrected freq
        Returns:
            feat: (B, D_stim)
        """
        if freqs.ndim == 1:
            freqs = freqs.unsqueeze(1)

        B = freqs.size(0)
        t = self.t.unsqueeze(0)

        harmonics = []
        for h in range(1, self.n_harmonics + 1):
            sin_h = torch.sin(2 * np.pi * h * freqs * t)  # (B, T)
            cos_h = torch.cos(2 * np.pi * h * freqs * t)  # (B, T)
            harmonics.append(sin_h)
            harmonics.append(cos_h)

        stim_harm = torch.stack(harmonics, dim=-1)        # (B, T, 2*n_harmonics)
        feat = self.encoder(stim_harm)                    # (B, hidden_dim)
        return feat


class StimulusBranchWithPhase(nn.Module):
    """
    Stimulus branch using StimulusEncoder with subject-specific phase correction.
    Input : labels (B,), phases (B,)
    Output: (B, D_stim)
    """
    def __init__(self, T, sfreq=250.0, hidden_dim=64, n_harmonics=3, out_dim=None):
        super().__init__()
        self.T = T
        self.sfreq = float(sfreq)
        self.n_harmonics = n_harmonics

        if out_dim is None:
            out_dim = hidden_dim

        # Stimulus encoder
        self.encoder = StimulusEncoder(in_dim=2 * n_harmonics, hidden_dim=hidden_dim)

        # Projection + normalization
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )
        self.out_dim = out_dim

        # Precompute time vector
        t = torch.arange(self.T, dtype=torch.float32) / np.float32(self.sfreq)
        self.register_buffer("t", t)  # (T,)

    def forward(self, freqs, phases):
        """
        Args:
            freqs:  (B,) frequencies (Hz)
            phases: (B,) phases (radian)
        Returns:
            feat: (B, out_dim)
        """
        device = freqs.device
        B = freqs.size(0)
        t = self.t.to(device).unsqueeze(0)  # (1, T)

        harmonics = []
        for h in range(1, self.n_harmonics + 1):
            # add phase shift per harmonic
            phase_term = phases.unsqueeze(1)  # (B, 1)
            sin_h = torch.sin(2 * np.pi * h * freqs.unsqueeze(1) * t + phase_term)
            cos_h = torch.cos(2 * np.pi * h * freqs.unsqueeze(1) * t + phase_term)
            harmonics.append(sin_h)
            harmonics.append(cos_h)

        stim_harm = torch.stack(harmonics, dim=-1).float()  # (B, T, 2*n_harmonics)

        feat = self.encoder(stim_harm)                      # (B, hidden_dim)
        feat = self.proj(feat)                              # (B, out_dim)
        return feat


class TemplateBranch(nn.Module):
    """
    Template branch using DTN.
    Input : (B, 1, C, T), labels (B,)
    Output : (B, D_temp) latent representation of the template
    """
    def __init__(self, n_bands, n_features, n_channels, n_samples, n_classes, D_temp=64):
        super().__init__()
        self.network = DTN(n_bands=n_bands, n_features=n_features,
                           n_channels=n_channels, n_samples=n_samples,
                           n_classes=n_classes)

        # Projection: (B, n_features) → (B, D_temp)
        self.proj = nn.Linear(n_features, D_temp)

    def forward(self, x, y=None, inference=False):
        """
        x: EEG input (Update template)
        y: Class index (Template index)
        """
        feat = self.network(x, y)
        feat = self.proj(feat)
        return feat
