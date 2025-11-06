# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import math


# ===== Base Simple Attention =====
class SimpleAttention(nn.Module):
    """
    Generic Simple Attention
    EEG feature = Key/Value  (B, T, D_eeg)
    Other feature (Stimulus or Template) = Query   (B, D_query)
    Output: logits (B, n_classes), attention feature (B, D_model)
    """
    def __init__(self, d_eeg, d_query, d_model, n_classes):
        super().__init__()
        self.key = nn.Linear(d_eeg, d_model)
        self.value = nn.Linear(d_eeg, d_model)
        self.query = nn.Linear(d_query, d_model)
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, eeg_feat, query_feat):
        """
        eeg_feat: (B, T, D_eeg)
        query_feat: (B, D_query)
        """
        K = self.key(eeg_feat)        # (B, T, D_model)
        V = self.value(eeg_feat)      # (B, T, D_model)
        Q = self.query(query_feat).unsqueeze(1)    # (B, T, D_model)

        # --- Attention computation ---
        # Dot-product similarity between Query and Key
        attn_score = (Q * K).sum(-1, keepdim=True) / math.sqrt(K.size(-1) + 1e-8)  # (B, T, 1)
        attn_weights = torch.sigmoid(attn_score + 0.1)  # stable gating between 0~1

        # --- Weighted sum over time dimension ---
        attn = (attn_weights * V).mean(dim=1)  # (B, D_model)

        # --- Classification ---
        logits = self.proj(attn)  # (B, n_classes)
        return logits, attn


# ===== EEG + Stimulus Attention =====
class SimpleAttention_EEG_Stimulus(SimpleAttention):
    def forward(self, eeg_feat, stim_feat):
        return super().forward(eeg_feat, stim_feat)


# ===== EEG + Template Attention =====
class SimpleAttention_EEG_Template(SimpleAttention):
    def forward(self, eeg_feat, temp_feat):
        return super().forward(eeg_feat, temp_feat)