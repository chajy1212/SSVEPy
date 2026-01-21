# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, chans, samples, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
        super().__init__()
        self.chans = chans
        self.samples = samples

        # First temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise spatial convolution
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropoutRate)

        # Separable convolution (temporal filtering)
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropoutRate)

        # Compute output dimensions dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)  # (B,1,C,T)
            out = self.forward_features(dummy)
            B, C, H, W = out.shape
            self.feature_dim = C
            self.sequence_len = W
            self.out_dim = C * W  # legacy flatten size


    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwiseConv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.drop1(x)
        x = self.separableConv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.drop2(x)
        return x                    # (B, F2, 1, W')


    def forward(self, x, return_sequence=False):
        """
        Args:
            x: (B, 1, chans, samples)
            return_sequence: if True, return (B, N, D) for attention
        """
        x = self.forward_features(x)  # (B, F2, 1, W')
        B, C, H, W = x.shape

        if return_sequence:
            # Convert (B, C, 1, W) → (B, W, C)
            x = x.squeeze(2).permute(0, 2, 1).contiguous()  # (B, N=W, D=C)
            return x  # sequence for attention: (B, N, D)
        else:
            # Legacy mode (for classifier)
            return x.view(B, -1)