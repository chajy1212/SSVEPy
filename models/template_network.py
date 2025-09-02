# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class FTN(nn.Module):
    """
    Fixed Template Network
     - Compares input EEG segments with fixed templates in a learned feature space.
     - Returns classification logits + latent representation
    """
    def __init__(self, n_channels, n_samples, n_classes, hidden_dim=64):
        super().__init__()
        self.n_classes = n_classes

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, (1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, hidden_dim, (n_channels, 1), bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Tanh()
        )

        # Fixed templates (frozen, not updated during training)
        self.register_buffer("templates", torch.randn(n_classes, 1, n_channels, n_samples))

        # Classification head
        self.fc = nn.Linear(hidden_dim, n_classes)


    def forward(self, x, return_feat=False):
        """
        Args:
            x (Tensor): EEG input of shape (B, 1, C, T)
            return_feat (bool):
                - True → return (logits, representation)
                - False → return logits only

        Returns:
            logits (Tensor): (B, n_classes) classification output
            feat_x (Tensor): (B, hidden_dim) latent representation
        """
        # EEG feature extraction
        feat_x = self.feature_extractor(x)          # (B, hidden_dim, 1, T')
        feat_x = feat_x.mean(dim=-1).squeeze(-1)    # (B, hidden_dim)

        # Template feature extraction
        t = self.feature_extractor(self.templates)  # (C, hidden_dim, 1, T')
        t = t.mean(dim=-1).squeeze(-1)              # (C, hidden_dim)

        # Cosine similarity between EEG features and templates
        sim = F.linear(F.normalize(feat_x, dim=1),
                       F.normalize(t, dim=1))       # (B, C)

        logits = self.fc(sim)

        if return_feat:
            return logits, feat_x
        return logits, feat_x


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
            x = torch.zeros(1, 1, n_channels, n_samples)
            feat = self.feature_extractor(x)
            self._register_templates(n_classes, *feat.shape[1:])


    def _register_templates(self, n_classes, *args):
        """ Initialize class-specific templates """
        self.register_buffer('running_template', torch.zeros(n_classes, *args))
        nn.init.xavier_uniform_(self.running_template, gain=1)


    def _update_templates(self, x, y):
        """ Update templates with exponential moving average (EMA) """
        with torch.no_grad():
            self.num_batches_tracked += 1
            factor = (1.0 / float(self.num_batches_tracked)) if self.momentum is None else self.momentum

            mask = F.one_hot(y, num_classes=self.running_template.shape[0]).float()  # (B, n_classes)
            features = self.feature_extractor(x)                                     # (B, F, C’, T’)

            mask_data = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * features.unsqueeze(1)
            new_template = mask_data.sum(0) / (mask.sum(0).view(-1, 1, 1, 1) + self.eps)

            self.running_template = (1 - factor) * self.running_template + factor * new_template


    def forward(self, x, y=None, return_feat=True):
        """
        x: (B, 1, C, T)
        return_feat=True → (logits=None, feat) 반환
        """
        x = self.instance_norm(x)
        feat = self.feature_extractor(x)        # (B, F, C’, T’)
        feat = self.global_pool(feat)           # (B, F, 1, 1)
        feat = feat.view(feat.size(0), -1)      # (B, F)
        feat = self.fc_drop(feat)

        # Update templates if training
        if self.training and y is not None:
            self._update_templates(x, y)

        if return_feat:
            return None, feat

        return None, feat
