# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, chans=64, samples=128,
                 dropout_rate=0.5, kern_length=64,
                 F1=8, D=2, F2=16):
        """
        Args:
            chans (int): number of EEG channels
            samples (int): number of time samples
            dropout_rate (float): dropout probability
            kern_length (int): temporal kernel length (usually sampling_rate/2)
            F1 (int): number of temporal filters
            D (int): depth multiplier
            F2 (int): number of pointwise filters (default F1*D)
        """
        super().__init__()

        # ----- Block 1: temporal convolution + depthwise spatial convolution -----
        self.block1 = nn.Sequential(
            # Temporal Conv (1 x kernLength)
            nn.Conv2d(1, F1, kernel_size=(1, kern_length),
                      padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),

            # Depthwise Conv across channels
            nn.Conv2d(F1, F1 * D, kernel_size=(chans, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),

            # Temporal downsampling
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout_rate)
        )

        # ----- Block 2: separable convolution (depthwise + pointwise) -----
        self.block2 = nn.Sequential(
            # Depthwise Conv (1 x 16)
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            # Pointwise Conv (1x1)
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),

            # Further temporal downsampling
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout_rate)
        )

        # Flatten layer → output (B, D)
        self.flatten = nn.Flatten()


    def forward(self, x):
        """
        Args:
            x: (B, 1, chans, samples)
        Returns:
            features: (B, D) flattened representation
        """
        x = self.block1(x)              # (B, F1*D, 1, T/4)
        x = self.block2(x)              # (B, F2, 1, T/32)
        x = self.flatten(x)             # (B, D)
        return x


class EEGNet_SSVEP(EEGNet):
    def __init__(self, chans=8, samples=256,
                 dropout_rate=0.5, kern_length=256,
                 F1=96, D=1, F2=96):
        super().__init__(chans, samples, dropout_rate, kern_length, F1, D, F2)