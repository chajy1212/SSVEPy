# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DTN(nn.Module):
    """
    Dynamic Template Network
     - Extracts conv features from EEG
     - Global average pooling for dimension reduction
     - Returns latent representation only
    """
    def __init__(self, n_bands, n_features, n_channels, n_samples, n_classes,
                 band_kernel=9, pooling_kernel=2,
                 dropout=0.5, momentum=None, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # One-hot table for template update
        self.register_buffer('encode_table', torch.eye(n_classes, dtype=torch.long))

        # Feature extractor
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('band_layer', nn.Conv2d(1, n_bands, (1, band_kernel),
                                     padding=(0, band_kernel // 2), bias=False)),
            ('spatial_layer', nn.Conv2d(n_bands, n_features, (n_channels, 1),
                                        bias=False)),
            ('temporal_layer1', nn.Conv2d(n_features, n_features, (1, pooling_kernel),
                                          stride=(1, pooling_kernel), bias=False)),
            ('bn', nn.BatchNorm2d(n_features)),
            ('tanh', nn.Tanh()),
            ('temporal_layer2', nn.Conv2d(n_features, n_features, (1, band_kernel),
                                          padding=(0, band_kernel // 2), bias=False)),
        ]))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_drop = nn.Dropout(dropout)
        self.instance_norm = nn.InstanceNorm2d(1)

        # Initialize running templates
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feat = self.feature_extractor(dummy) # (1, F, H, W)
            self.register_buffer('running_template', torch.zeros(n_classes, *feat.shape[1:]))
            nn.init.xavier_uniform_(self.running_template, gain=1)


    def _update_templates(self, features, y):
        """ Update templates with exponential moving average (EMA) """
        with torch.no_grad():
            self.num_batches_tracked += 1
            factor = (1.0 / float(self.num_batches_tracked)) if self.momentum is None else self.momentum

            mask = F.one_hot(y, num_classes=self.running_template.shape[0]).float()     # (B, n_classes)

            # features: (B, F, H, W)
            # mask: (B, C) -> (B, C, 1, 1, 1)
            mask_data = mask.view(*mask.shape, 1, 1, 1) * features.unsqueeze(1)         # (B, C, F, H, W)

            sum_features = mask_data.sum(0)  # (C, F, H, W)
            sum_counts = mask.sum(0).view(-1, 1, 1, 1) + self.eps
            new_template_batch = sum_features / sum_counts
            update_idx = (mask.sum(0) > 0)

            self.running_template[update_idx] = (1 - factor) * self.running_template[update_idx] \
                                                + factor * new_template_batch[update_idx]


    def forward(self, x, y=None):
        x = self.instance_norm(x)
        features = self.feature_extractor(x)

        if self.training and y is not None:
            self._update_templates(features, y)

        if y is not None:
            out_feat = self.running_template[y]
        else:
            out_feat = features

        out_feat = self.global_pool(out_feat)
        out_feat = out_feat.view(out_feat.size(0), -1)
        out_feat = self.fc_drop(out_feat)

        return out_feat