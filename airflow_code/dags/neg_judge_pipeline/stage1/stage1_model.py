import torch
import torch.nn as nn
import torchvision.models as models
from .stage1_config import *

class MultiFrameClassifier(nn.Module):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        frames_per_sample=FRAMES_PER_SAMPLE,
        dropout_rate=0.39,
        hidden_dim=256
    ):
        super().__init__()
        self.frames_per_sample = frames_per_sample

        # ResNet34 feature extractor (ImageNet pretrained)
        resnet = models.resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        # Final classifier with tunable hidden dimension and dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, frames):
        feats = []
        for x in frames:
            feat = self.feature_extractor(x)  # [B, 512, 1, 1]
            feat = feat.flatten(1)            # [B, 512]
            feats.append(feat)
        # Temporal average pooling
        pooled_feats = torch.mean(torch.stack(feats, dim=1), dim=1)  # [B, 512]

        # Ensure batch dim for single sample
        if pooled_feats.dim() == 1:
            pooled_feats = pooled_feats.unsqueeze(0)

        out = self.classifier(pooled_feats)
        return out
