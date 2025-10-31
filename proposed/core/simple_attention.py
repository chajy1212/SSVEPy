# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np


# ===== Base Simple Attention =====
class SimpleAttention(nn.Module):
    """
    Generic Simple Attention
    EEG feature = Key/Value
    Other feature (Stimulus or Template) = Query
    """
    def __init__(self, d_eeg, d_query, d_model, n_classes):
        super().__init__()
        self.key = nn.Linear(d_eeg, d_model)
        self.value = nn.Linear(d_eeg, d_model)
        self.query = nn.Linear(d_query, d_model)
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, eeg_feat, query_feat):
        """
        eeg_feat:   (B, D_eeg)
        query_feat: (B, D_query)  -> Stimulus or Template
        """
        K = self.key(eeg_feat)        # (B, D_model)
        V = self.value(eeg_feat)      # (B, D_model)
        Q = self.query(query_feat)    # (B, D_model)

        attn_score = (Q * K).sum(-1, keepdim=True) / np.sqrt(K.size(-1))  # (B, 1)
        attn_weights = torch.sigmoid(attn_score)                          # (B, 1)

        attn = attn_weights * V   # (B, D_model)
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



