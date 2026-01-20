# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class StimulusAutoCorrector(nn.Module):
    def __init__(self, eeg_channels, hidden_dim=64):
        super().__init__()

        # Simple CNN to extract features from EEG input
        self.eeg_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(eeg_channels, 1), stride=1),
            nn.ReLU(),
        )

        # Regressor to predict Δf
        # Input: EEG Features (32) + Nominal Freq (1)
        # Output: Δf (Hz)
        self.regressor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Projection Heads for SSL Loss (Lazy Initialization)
        self.proj_eeg = None
        self.proj_stim = None

    def build_projection(self, eeg_dim, stim_dim, device):
        """
        Initializes projection layers dynamically based on input dimensions.
        """
        self.proj_eeg = nn.Linear(eeg_dim, 128).to(device)
        self.proj_stim = nn.Linear(stim_dim, 128).to(device)

    def forward(self, eeg, nominal_freq):
        """
        Args:
            eeg: (B, 1, C, T) - Raw EEG input
            nominal_freq: (B,) - Nominal target frequencies (Hz)
        Returns:
            corrected_freq: (B,) - Adjusted frequencies
            delta_f: (B,) - Predicted frequency deviations
        """
        # CNN Feature Extraction
        feat = self.eeg_encoder(eeg)  # (B, 32, 1, T')
        feat = feat.mean(dim=[2, 3])  # Global Pooling -> (B, 32)

        # Concatenate with Nominal Frequency
        nominal = nominal_freq.view(-1, 1).float()
        x = torch.cat([feat, nominal], dim=1)

        # Predict Delta f
        delta_f = self.regressor(x).squeeze(1)  # (B,)

        # Apply Correction
        corrected_freq = nominal_freq + delta_f

        return corrected_freq, delta_f

    def compute_ssl_loss(self, eeg_feat, stim_feat):
        """
        Computes Self-Supervised Learning (SSL) loss to align EEG and Stimulus features.

        Args:
            eeg_feat: (B, T, D) or (B, D) - Output from EEGBranch
            stim_feat: (B, D) - Output from StimulusBranch
        """
        device = eeg_feat.device

        # Perform Global Pooling if EEG Feature is a sequence
        if eeg_feat.dim() == 3:
            eeg_feat_pooled = eeg_feat.mean(dim=1)  # (B, D)
        else:
            eeg_feat_pooled = eeg_feat

        # Create Projection Layer if not exists (First run)
        if self.proj_eeg is None:
            self.build_projection(eeg_feat_pooled.shape[1], stim_feat.shape[1], device)

        # Projection to common space
        z_eeg = self.proj_eeg(eeg_feat_pooled)
        z_stim = self.proj_stim(stim_feat)

        # Cosine Similarity Maximization (Loss is negative similarity)
        cos_sim = F.cosine_similarity(z_eeg, z_stim, dim=1)
        loss = -cos_sim.mean()

        return loss