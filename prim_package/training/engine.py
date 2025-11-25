from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .losses import ContrastiveLoss, TripletLoss, pairwise_distance


def train_contrastive(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float = 1.0,
    distance: str = "cosine",
) -> float:
    """
    Train one epoch with contrastive loss.
    """
    model.train()
    criterion = ContrastiveLoss(margin=margin, distance=distance)
    total_loss = 0.0

    for img1, img2, y in loader:
        img1, img2, y = img1.to(device), img2.to(device), y.to(device)

        z1, z2 = model(img1, img2)
        loss = criterion(z1, z2, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img1.size(0)

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def validate_contrastive(
    model,
    loader: DataLoader,
    device: torch.device,
    distance: str = "cosine",
) -> Dict[str, float]:
    """
    Validate model by reporting mean distances for positive and negative pairs.
    """
    model.eval()
    pos_dists: List[float] = []
    neg_dists: List[float] = []

    for img1, img2, y in loader:
        img1, img2 = img1.to(device), img2.to(device)
        z1, z2 = model(img1, img2)
        d = pairwise_distance(z1, z2, mode=distance).cpu()

        for di, yi in zip(d, y):
            if float(yi) == 1.0:
                pos_dists.append(float(di))
            else:
                neg_dists.append(float(di))

    pos_mean = sum(pos_dists) / max(1, len(pos_dists))
    neg_mean = sum(neg_dists) / max(1, len(neg_dists))
    return {"pos_mean": pos_mean, "neg_mean": neg_mean}


def train_triplet(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float = 0.3,
    distance: str = "cosine",
) -> float:
    """
    Train one epoch with triplet loss.
    """
    model.train()
    criterion = TripletLoss(margin=margin, distance=distance)
    total_loss = 0.0

    for a, p, n in loader:
        a, p, n = a.to(device), p.to(device), n.to(device)

        # Reuse shared encoder via SiameseNet API
        za = model.forward_once(a)
        zp = model.forward_once(p)
        zn = model.forward_once(n)

        loss = criterion(za, zp, zn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * a.size(0)

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def validate_triplet(
    model,
    loader: DataLoader,
    device: torch.device,
    distance: str = "cosine",
) -> Dict[str, float]:
    """
    Validate model by reporting mean d(a,p) and d(a,n).
    """
    model.eval()
    d_ap_all: List[float] = []
    d_an_all: List[float] = []

    for a, p, n in loader:
        a, p, n = a.to(device), p.to(device), n.to(device)

        za = model.forward_once(a)
        zp = model.forward_once(p)
        zn = model.forward_once(n)

        d_ap = pairwise_distance(za, zp, mode=distance).cpu().tolist()
        d_an = pairwise_distance(za, zn, mode=distance).cpu().tolist()

        d_ap_all.extend([float(x) for x in d_ap])
        d_an_all.extend([float(x) for x in d_an])

    ap_mean = sum(d_ap_all) / max(1, len(d_ap_all))
    an_mean = sum(d_an_all) / max(1, len(d_an_all))
    return {"ap_mean": ap_mean, "an_mean": an_mean}