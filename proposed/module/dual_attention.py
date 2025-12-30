# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding


class DualAttention(nn.Module):
    def __init__(self, d_eeg, d_query, d_model, num_heads, proj_dim, entropy_lambda=1e-4):
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
        self.entropy_lambda = entropy_lambda

        self.eeg_proj = nn.Linear(d_eeg, d_model)
        self.stim_proj = nn.Linear(d_query, d_model)
        self.temp_proj = nn.Linear(d_query, d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, proj_dim)

        self.rotary_emb = RotaryEmbedding(dim=self.d_head)

        # normalization layers
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)


    def forward(self, eeg_feat, stim_feat, temp_feat):
        """
        eeg_feat: (B, N, D_eeg)  <-- multi-key EEG feature sequence
        stim_feat: (B, D_query)
        temp_feat: (B, D_query)
        """
        B, N, _ = eeg_feat.shape

        # Linear projections
        eeg = self.eeg_proj(eeg_feat)                  # (B, N, D)
        stim = self.stim_proj(stim_feat).unsqueeze(1)  # (B, 1, D)
        temp = self.temp_proj(temp_feat).unsqueeze(1)  # (B, 1, D)

        # concat [stim, temp] as queries
        query = torch.cat([stim, temp], dim=1)      # (B, 2, D)
        key_value = eeg                             # (B, N, D)

        # Normalization
        query = self.norm_q(query)
        key_value = self.norm_k(key_value)

        # Q, K, V projection
        Q = self.q_proj(query)  # (B, 2, D)
        K = self.k_proj(key_value)  # (B, N, D)
        V = self.v_proj(key_value)  # (B, N, D)

        # Split heads
        Q = Q.view(B, 2, self.num_heads, self.d_head)
        K = K.view(B, N, self.num_heads, self.d_head)
        V = V.view(B, N, self.num_heads, self.d_head)

        # Rotary embeddings
        Q = self.rotary_emb.rotate_queries_or_keys(Q)
        K = self.rotary_emb.rotate_queries_or_keys(K)

        # Normalize Q/K (unit sphere)
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        # Scaled dot-product attention
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) / math.sqrt(self.d_head)
        scale = attn_scores.std(dim=-1, keepdim=True).clamp(min=1e-3).detach()
        attn_scores = torch.tanh(attn_scores / (2.0 * scale)) * (5.0 * scale)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = (attn_weights + 1e-4) / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Entropy for monitoring and regularization
        attn_entropy = -(attn_weights * (attn_weights + 1e-9).log()).sum(dim=-1).mean()
        if self.training and self.entropy_lambda > 0:
            self.loss_entropy = -self.entropy_lambda * attn_entropy
        else:
            self.loss_entropy = torch.tensor(0.0, device=eeg_feat.device)

        # Attention output
        attn_out = torch.einsum("bhqk,bkhd->bqhd", attn_weights, V)
        attn_out = attn_out.contiguous().view(B, attn_out.size(1), -1)  # (B,2,D)

        # Pool stim+temp attended outputs
        pooled = attn_out.mean(dim=1)
        logits = self.fc(pooled)

        # Debug
        # if not self.training:
        #     print(f"[DEBUG] Entropy: {attn_entropy.item():.3f}, Scores std: {attn_scores.std().item():.3f}")

        return logits, pooled, attn_out, query