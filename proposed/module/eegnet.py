# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, chans, dropoutRate=0.5, kernLength=256, F1=96, D=1, F2=96):
        """
        :param dropoutRate: Dropout fraction to prevent overfitting
        :param kernLength: Length of the temporal convolution kernel
        :param F1: Number of temporal filters
        :param D: Number of spatial filters to learn within each temporal convolution
        :param F2: Number of pointwise filters (final feature maps)
        """
        super().__init__()
        self.chans = chans

        # Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise spatial convolution
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropoutRate)

        # Separable convolution (Temporal + Pointwise)
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropoutRate)


    def forward(self, x):
        """
        Input: x (B, 1, C, T)
        Output: x (B, 96, 1, T')    # T' is the compressed time-step after pooling operations
        """
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

        return x                    # (B, F2, 1, T')