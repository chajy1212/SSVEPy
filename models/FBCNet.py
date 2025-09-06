# -*- coding:utf-8 -*-
import sys
import torch
import torch.nn as nn


# Constraint layers
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)


# Temporal aggregator layers
class LogVarLayer(nn.Module):
    """ The log variance layer: calculates the log variance of the data along given 'dim' """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


# Activation
class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FBCNet(nn.Module):
    """
    FBNet with seperate variance for every 1s.
    Input shape: (B, 1, C, T, Bands)
    Output shape: (B, feature_dim)
    m : number of sptatial filters
    nBands: number of bands in the data
    """
    def __init__(self, nChan, nBands=9, m=32, strideFactor=4, temporalLayer='LogVarLayer', doWeightNorm=True):
        super().__init__()
        self.nBands = 1
        self.m = m
        self.strideFactor = strideFactor

        # Spatial Convolution Block
        self.scb = nn.Sequential(
            Conv2dWithConstraint(nBands, m * nBands, (nChan, 1), groups=nBands,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(m * nBands),
            swish()
        )

        # Temporal aggregator
        self.temporalLayer = getattr(sys.modules[__name__], temporalLayer)(dim=3)

        # Output feature dimension
        self.out_dim = m * nBands * strideFactor


    def forward(self, x):
        x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)    # (B, 1, C, T, Bands) → (B, Bands, C, T)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim=1)                       # (B, out_dim)
        return x