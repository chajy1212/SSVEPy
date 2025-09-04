# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

class DualAttention(nn.Module):
    """
    Dual Attention module
    - EEG → Key/Value
    - Stimulus & Template → Query
    - Attention outputs (A_stim, A_temp) fused by concatenation + depthwise conv
    - Projection head for SSL (self-supervised learning)
    """
    def __init__(self, d_eeg, d_query, d_model, num_heads=4, proj_dim=64):
        """
        Args:
            d_eeg (int): EEG feature dimension (flattened from EEG encoder)
            d_query (int): Query feature dimension (Stimulus/Template encoder output)
            d_model (int): Transformer hidden dimension
            num_heads (int): number of attention heads
            proj_dim (int): final projection dimension
        """
        super().__init__()

        # EEG → Key/Value
        self.key = nn.Linear(d_eeg, d_model)
        self.value = nn.Linear(d_eeg, d_model)

        # Stimulus/Template → Query
        self.query_temp = nn.Linear(d_query, d_model)
        self.query_stim = nn.Linear(d_query, d_model)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        # Fusion: [A_temp, A_stim] concat + depthwise conv
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(2 * d_model, 2 * d_model, kernel_size=3, padding=1, groups=2 * d_model),  # depthwise
            nn.ReLU(),
            nn.Conv1d(2 * d_model, d_model, kernel_size=1)  # pointwise
        )

        # Projection head (MLP)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )


    def forward(self, eeg_feat, stim_feat, temp_feat):
        """
        Args:
            eeg_feat  : (B, D_eeg) from EEG encoder
            stim_feat : (B, D_query) from Stimulus encoder
            temp_feat : (B, D_query) from Template encoder

        Returns:
            proj      : (B, proj_dim) final representation
            A_stim    : (B, d_model) stimulus attention output
            A_temp    : (B, d_model) template attention output
        """
        # EEG → Key/Value
        K = self.key(eeg_feat).unsqueeze(1)  # (B, 1, d_model)
        V = self.value(eeg_feat).unsqueeze(1)  # (B, 1, d_model)

        # Stimulus / Template → Query
        Q_stim = self.query_stim(stim_feat).unsqueeze(1)  # (B, 1, d_model)
        Q_temp = self.query_temp(temp_feat).unsqueeze(1)  # (B, 1, d_model)

        # Attention
        A_stim, _ = self.attn(Q_stim, K, V)  # (B, 1, d_model)
        A_temp, _ = self.attn(Q_temp, K, V)  # (B, 1, d_model)

        # Fusion: concat along channel → depthwise + pointwise conv
        A_cat = torch.cat([A_stim, A_temp], dim=-1)  # (B, 1, 2*d_model)
        A_cat = A_cat.permute(0, 2, 1)  # (B, 2*d_model, 1)
        A_fused = self.fusion_conv(A_cat)  # (B, d_model, 1)
        A_fused = A_fused.squeeze(-1)  # (B, d_model)

        # Projection head
        proj = self.proj_head(A_fused)  # (B, proj_dim)

        return proj, A_stim.squeeze(1), A_temp.squeeze(1)