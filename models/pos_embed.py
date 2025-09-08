# -*- coding:utf-8 -*-
import torch
import math
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Vaswani et al., 2017)
    Adds deterministic sine & cosine positional encodings to input embeddings.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model (int): Embedding dimension
        max_len (int): Maximum sequence length to precompute (default=5000)

    Input: (B, N, D) → sequence length N, embedding dim D
            B = batch size
            N = sequence length
            D = embedding dimension

    Output: (B, N, D) + positional encoding
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Precompute PE matrix
        pe = torch.zeros(max_len, d_model)                                   # (max_len, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, D)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input embeddings, shape (B, N, D)

        Returns:
            torch.Tensor: Positional-encoded embeddings, shape (B, N, D)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]