# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding


class DualAttention(nn.Module):
    def __init__(self, d_eeg, d_query, d_model, num_heads, proj_dim):
        """
        Dual Attention with Rotary Positional Embedding (RoPE)
        - eeg_feat: EEG feature vector
        - stim_feat: Stimulus reference feature
        - temp_feat: Template feature
        """
        super().__init__()
        self.eeg_proj = nn.Linear(d_eeg, d_model)
        self.stim_proj = nn.Linear(d_query, d_model)
        self.temp_proj = nn.Linear(d_query, d_model)

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(d_model, proj_dim)

        # rotary positional encoding
        self.rotary_emb = RotaryEmbedding(dim=d_model // num_heads)


    def forward(self, eeg_feat, stim_feat, temp_feat):
        """
        eeg_feat: (B, D_eeg)
        stim_feat: (B, D_query)
        temp_feat: (B, D_query)
        """
        B = eeg_feat.size(0)

        eeg = self.eeg_proj(eeg_feat).unsqueeze(1)      # (B, 1, D)
        stim = self.stim_proj(stim_feat).unsqueeze(1)   # (B, 1, D)
        temp = self.temp_proj(temp_feat).unsqueeze(1)   # (B, 1, D)

        # concat sequence: [stim, temp] as queries, eeg as key/value
        query = torch.cat([stim, temp], dim=1)      # (B, 2, D)
        key_value = eeg                         # (B, 1, D)

        # apply RoPE to Q and K
        q_rot = self.attn.in_proj_weight[: self.attn.embed_dim]                         # Q projection weight
        k_rot = self.attn.in_proj_weight[self.attn.embed_dim: 2 * self.attn.embed_dim]  # K projection weight

        # project queries & keys
        Q = torch.matmul(query, q_rot.T)      # (B, 2, D)
        K = torch.matmul(key_value, k_rot.T)  # (B, 1, D)

        # split heads
        Q = Q.view(B, Q.size(1), self.attn.num_heads, -1)  # (B, 2, H, D_head)
        K = K.view(B, K.size(1), self.attn.num_heads, -1)  # (B, 1, H, D_head), D_head = D / num_heads

        # apply rotary embedding (adds relative position info)
        Q, K = self.rotary_emb.rotate_queries_or_keys(Q), self.rotary_emb.rotate_queries_or_keys(K)

        # reshape back
        Q = Q.view(B, Q.size(1), -1)
        K = K.view(B, K.size(1), -1)

        # attention
        attn_out, _ = self.attn(Q, K, key_value)  # (B, 2, D)

        # use both stim+temp attended outputs → pool
        pooled = attn_out.mean(dim=1)             # (B, D)

        logits = self.fc(pooled)                  # (B, proj_dim)

        return logits, attn_out, query