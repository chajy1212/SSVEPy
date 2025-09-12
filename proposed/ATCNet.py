# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self, chans, samples, F1=16, D=2, kernel_size=64, pool_size=7, dropout=0.3):
        super().__init__()
        F2 = F1 * D

        self.temporal_conv = nn.Conv2d(1, F1, (1, kernel_size), padding=(0, kernel_size // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)

        self.pointwise_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)

        self.pool1 = nn.AvgPool2d((1, 8))
        self.pool2 = nn.AvgPool2d((1, pool_size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.bn1(x)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.pointwise_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        # (B, F2, 1, T') -> (B, T', F2)
        x = x.squeeze(2).permute(0, 2, 1)
        return x


class TCN_Block(nn.Module):
    def __init__(self, input_dim, filters=32, kernel_size=4, depth=2, dropout=0.3, activation='elu'):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.activation = getattr(F, activation) if hasattr(F, activation) else F.elu

        for i in range(depth):
            dilation = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(input_dim, filters, kernel_size,
                          padding=dilation * (kernel_size - 1),
                          dilation=dilation),
                nn.BatchNorm1d(filters),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Conv1d(filters, filters, kernel_size,
                          padding=dilation * (kernel_size - 1),
                          dilation=dilation),
                nn.BatchNorm1d(filters),
                nn.ELU(),
                nn.Dropout(dropout)
            ))
            input_dim = filters

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (B, F, T)
        for block in self.blocks:
            out = block(x)
            out = out[..., :x.size(2)]  # Remove causal padding
            x = out + x
        return x.permute(0, 2, 1)       # (B, T, F)


class ATCNet(nn.Module):
    def __init__(self, chans, samples,
                 F1=16, D=2, kernel_size=64, pool_size=7, dropout=0.3,
                 n_windows=5, d_model=32, n_heads=4,
                 tcn_depth=2, tcn_kernel_size=4, tcn_filters=32, tcn_dropout=0.3):
        super().__init__()
        self.conv_block = Conv_Block(chans, samples, F1, D, kernel_size, pool_size, dropout)
        self.n_windows = n_windows

        self.layernorm = nn.LayerNorm(F1 * D)
        self.attn = nn.MultiheadAttention(embed_dim=F1 * D, num_heads=n_heads, batch_first=True)

        self.tcn = TCN_Block(F1 * D, filters=tcn_filters, kernel_size=tcn_kernel_size,
                                 depth=tcn_depth, dropout=tcn_dropout)

    def forward(self, x):
        x = self.conv_block(x)                      # (B, T', F)
        B, T, Fdim = x.size()

        window_size = T // self.n_windows
        outputs = []
        for i in range(self.n_windows):
            start = i * window_size
            end = start + window_size
            win = x[:, start:end, :]                # (B, win, F)

            win = self.layernorm(win)
            attn_out, _ = self.attn(win, win, win)  # (B, win, F)

            tcn_out = self.tcn(attn_out)            # (B, win, F)
            feat = tcn_out[:, -1, :]
            outputs.append(feat)

        out = torch.cat(outputs, dim=1)             # (B, F * n_windows)
        return out