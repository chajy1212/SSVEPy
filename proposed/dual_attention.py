# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        self.eeg_proj = nn.Linear(d_eeg, d_model)
        self.stim_proj = nn.Linear(d_query, d_model)
        self.temp_proj = nn.Linear(d_query, d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, proj_dim)

        # rotary positional encoding
        self.rotary_emb = RotaryEmbedding(dim=self.d_head)


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
        key_value = eeg                             # (B, 1, D)

        # Q, K, V projection
        Q = self.q_proj(query)      # (B, 2, D)
        K = self.k_proj(key_value)  # (B, 1, D)
        V = self.v_proj(key_value)  # (B, 1, D)

        # split heads
        Q = Q.view(B, Q.size(1), self.num_heads, self.d_head)  # (B, 2, H, D_head)
        K = K.view(B, K.size(1), self.num_heads, self.d_head)  # (B, 1, H, D_head)
        V = V.view(B, V.size(1), self.num_heads, self.d_head)  # (B, 1, H, D_head)

        # apply rotary embedding
        Q = self.rotary_emb.rotate_queries_or_keys(Q)
        K = self.rotary_emb.rotate_queries_or_keys(K)

        # scaled dot-product attention
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) / (self.d_head ** 0.5)  # (B, H, 2, 1)
        attn_weights = F.softmax(attn_scores, dim=-1)                                # (B, H, 2, 1)
        attn_out = torch.einsum("bhqk,bkhd->bqhd", attn_weights, V)                 # (B, 2, H, D_head)

        # reshape back
        Q = Q.view(B, Q.size(1), -1)
        K = K.view(B, K.size(1), -1)

        # reshape back
        attn_out = attn_out.contiguous().view(B, attn_out.size(1), -1)  # (B,2,D)

        # use both stim+temp attended outputs → pool
        pooled = attn_out.mean(dim=1)             # (B, D)

        logits = self.fc(pooled)                  # (B, proj_dim)

        return logits, pooled, attn_out, query