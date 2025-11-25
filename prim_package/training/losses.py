from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance(z1: torch.Tensor, z2: torch.Tensor, mode: str = "cosine") -> torch.Tensor:
    """
    Compute pairwise distance between two embedding batches.

    Args:
        z1: Tensor of shape (B, D)
        z2: Tensor of shape (B, D)
        mode: "cosine" (distance in [0, 2]) or "euclidean"

    Returns:
        Tensor of shape (B,) with distances
    """
    if mode == "cosine":
        # return distance in [0, 2], lower is more similar
        sim = F.cosine_similarity(z1, z2)
        return 1.0 - sim
    elif mode == "euclidean":
        return torch.sqrt(torch.sum((z1 - z2) ** 2, dim=1) + 1e-9)
    else:
        raise ValueError(f"Unknown distance mode '{mode}'. Use 'cosine' or 'euclidean'.")


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for pair labels y in {0,1}:
      L = y * d^2 + (1-y) * max(0, margin - d)^2

    Where d is distance between embeddings.
    """

    def __init__(self, margin: float = 1.0, distance: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d = pairwise_distance(z1, z2, mode=self.distance)
        pos = y * (d ** 2)
        neg = (1.0 - y) * (F.relu(self.margin - d) ** 2)
        return torch.mean(pos + neg)


class TripletLoss(nn.Module):
    """
    Triplet margin loss:
      L = max(0, d(a, p) - d(a, n) + margin)

    Using either cosine or euclidean distance.
    """

    def __init__(self, margin: float = 0.3, distance: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        d_ap = pairwise_distance(a, p, mode=self.distance)
        d_an = pairwise_distance(a, n, mode=self.distance)
        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()