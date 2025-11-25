from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    """
    ResNet50 backbone (feature extractor) + MLP projection head.
    Output embeddings are L2-normalized.
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        # TorchVision ResNet50; use weights arg for >=0.13 API
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        # Remove classifier head
        self.backbone.fc = nn.Identity()
        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)


class SiameseNet(nn.Module):
    """
    Siamese wrapper around the shared EmbeddingNet encoder.
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.encoder = EmbeddingNet(embed_dim=embed_dim, pretrained=pretrained)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2