# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from pos_embed import SinusoidalPositionalEncoding


class DualAttention(nn.Module):
    def __init__(self, d_eeg, d_query, d_model, num_heads, proj_dim):
        super().__init__()
        self.eeg_proj = nn.Linear(d_eeg, d_model)
        self.stim_proj = nn.Linear(d_query, d_model)
        self.temp_proj = nn.Linear(d_query, d_model)

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(d_model, proj_dim)

        # positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)

    def forward(self, eeg_feat, stim_feat, temp_feat):
        """
        eeg_feat: (B, D_eeg)
        stim_feat: (B, D_query)
        temp_feat: (B, D_query)
        """
        # (B, 1, D)
        eeg = self.eeg_proj(eeg_feat).unsqueeze(1)
        stim = self.stim_proj(stim_feat).unsqueeze(1)
        temp = self.temp_proj(temp_feat).unsqueeze(1)

        # concat sequence: [stim, temp] as queries, eeg as key/value
        query = torch.cat([stim, temp], dim=1)  # (B, 2, D)
        key_value = eeg                                # (B, 1, D)

        # positional encoding
        query = self.pos_encoding(query)
        key_value = self.pos_encoding(key_value)

        # attention
        attn_out, _ = self.attn(query, key_value, key_value)  # (B, 2, D)

        # use both stim+temp attended outputs → pool
        pooled = attn_out.mean(dim=1)  # (B, D)

        logits = self.fc(pooled)       # (B, proj_dim)

        return logits, attn_out, query