# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import math


def generate_stimulus_signal(freq, T, sfreq):
    B = freq.size(0)
    t = torch.linspace(0, T / sfreq, T, device=freq.device).unsqueeze(0)  # (1, T)

    freq = freq.unsqueeze(1)  # (B, 1)

    stim_sin = torch.sin(2 * math.pi * freq * t)  # (B, T)
    stim_cos = torch.cos(2 * math.pi * freq * t)

    stim = torch.stack([stim_sin, stim_cos], dim=2)  # (B, T, 2)

    return stim


class StimulusEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # preserves temporal structure and frequency information
        self.encoder = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # global temporal pooling → single embedding
        )

    def forward(self, stim):
        stim = stim.permute(0, 2, 1)    # (B, T, 2) -> (B, 2, T)
        feat = self.encoder(stim)       # (B, hidden_dim, 1)
        feat = feat.squeeze(-1)         # (B, hidden_dim)

        return feat