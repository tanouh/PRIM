from __future__ import annotations

from typing import Dict, List, Optional
import os
import csv

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
    save_csv_path: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Validate model by:
      - reporting mean distances for positive and negative pairs
      - optionally saving per-pair predictions to CSV.

    CSV columns (when save_csv_path is provided):
      - a_image_path, b_image_path, distance, label_true
      - pred, decision, threshold  (only if threshold is provided)

    Notes:
      - Lower distance means more similar for both cosine and euclidean modes.
      - If 'threshold' is provided, pred = 1 (similar) if distance < threshold else 0.
    """
    model.eval()
    pos_dists: List[float] = []
    neg_dists: List[float] = []

    # If saving CSV, buffer rows in memory then write once at end
    buffer_rows: List[List[str]] = []
    have_paths = False  # will switch to True if the loader provides paths

    for batch in loader:
        # Support (img1, img2, y) or (img1, img2, y, p1, p2) where p1/p2 are lists of paths
        if isinstance(batch, (list, tuple)) and len(batch) == 5:
            img1, img2, y, p1_list, p2_list = batch
            have_paths = True
        else:
            img1, img2, y = batch
            p1_list, p2_list = None, None

        img1, img2 = img1.to(device), img2.to(device)
        z1, z2 = model(img1, img2)
        d = pairwise_distance(z1, z2, mode=distance).cpu()  # shape (B,)

        # y is typically a 1D tensor length B
        for i in range(d.shape[0]):
            di = float(d[i])
            yi = float(y[i])

            if yi == 1.0:
                pos_dists.append(di)
            else:
                neg_dists.append(di)

            if save_csv_path is not None:
                a_path = p1_list[i] if (have_paths and p1_list is not None) else ""
                b_path = p2_list[i] if (have_paths and p2_list is not None) else ""
                if threshold is not None:
                    pred = 1 if di < threshold else 0
                    decision = "similar" if pred == 1 else "dissimilar"
                    buffer_rows.append([
                        str(a_path),
                        str(b_path),
                        f"{di:.8f}",
                        str(int(yi)),
                        str(int(pred)),
                        decision,
                        str(threshold),
                    ])
                else:
                    # No threshold: save distances and labels only
                    buffer_rows.append([
                        str(a_path),
                        str(b_path),
                        f"{di:.8f}",
                        str(int(yi)),
                    ])

    pos_mean = sum(pos_dists) / max(1, len(pos_dists))
    neg_mean = sum(neg_dists) / max(1, len(neg_dists))

    # Write CSV if requested
    if save_csv_path is not None:
        out_dir = os.path.dirname(save_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Header depends on threshold presence
        if threshold is not None:
            header = ["a_image_path", "b_image_path", "distance", "label_true", "pred", "decision", "threshold"]
        else:
            header = ["a_image_path", "b_image_path", "distance", "label_true"]

        with open(save_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(buffer_rows)

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
    save_csv_path: Optional[str] = None,
    ap_threshold: Optional[float] = None,
    an_threshold: Optional[float] = None,
    delta_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Validate model by reporting mean d(a,p) and d(a,n), and optionally save per-triplet logs.

    If 'save_csv_path' is provided, a CSV will be written with:
      - anchor_path, pos_path, neg_path, d_ap, d_an, delta
      - If thresholds are provided, the following columns are appended:
          * when ap_threshold:    ap_threshold, ap_pred, ap_decision
          * when an_threshold:    an_threshold, an_pred, an_decision
          * when delta_threshold: delta_threshold, delta_pred, delta_decision

    Conventions:
      - Lower distance => more similar (for both cosine and euclidean).
      - ap_pred = 1 (similar) if d_ap < ap_threshold else 0
      - an_pred = 1 (dissimilar) if d_an > an_threshold else 0
      - delta = d_an - d_ap; delta_pred = 1 (correct ordering) if delta > delta_threshold else 0
    """
    model.eval()
    d_ap_all: List[float] = []
    d_an_all: List[float] = []

    buffer_rows: List[List[str]] = []
    have_paths = False

    for batch in loader:
        # Support (a,p,n) or (a,p,n,a_path,p_path,n_path)
        if isinstance(batch, (list, tuple)) and len(batch) == 6:
            a, p, n, a_paths, p_paths, n_paths = batch
            have_paths = True
        else:
            a, p, n = batch
            a_paths = p_paths = n_paths = None

        a, p, n = a.to(device), p.to(device), n.to(device)

        za = model.forward_once(a)
        zp = model.forward_once(p)
        zn = model.forward_once(n)

        d_ap = pairwise_distance(za, zp, mode=distance).cpu().tolist()
        d_an = pairwise_distance(za, zn, mode=distance).cpu().tolist()

        for i in range(len(d_ap)):
            dap = float(d_ap[i])
            dan = float(d_an[i])
            delta = dan - dap

            d_ap_all.append(dap)
            d_an_all.append(dan)

            if save_csv_path is not None:
                a_path = a_paths[i] if (have_paths and a_paths is not None) else ""
                p_path = p_paths[i] if (have_paths and p_paths is not None) else ""
                n_path = n_paths[i] if (have_paths and n_paths is not None) else ""

                row = [
                    str(a_path),
                    str(p_path),
                    str(n_path),
                    f"{dap:.8f}",
                    f"{dan:.8f}",
                    f"{delta:.8f}",
                ]

                # ap decision
                if ap_threshold is not None:
                    ap_pred = 1 if dap < ap_threshold else 0
                    ap_decision = "anchor~pos similar" if ap_pred == 1 else "not similar"
                    row.extend([str(ap_threshold), str(int(ap_pred)), ap_decision])

                # an decision
                if an_threshold is not None:
                    an_pred = 1 if dan > an_threshold else 0
                    an_decision = "anchor~neg dissimilar" if an_pred == 1 else "not dissimilar"
                    row.extend([str(an_threshold), str(int(an_pred)), an_decision])

                # delta decision
                if delta_threshold is not None:
                    delta_pred = 1 if delta > delta_threshold else 0
                    delta_decision = "delta ok (d_an-d_ap large)" if delta_pred == 1 else "delta small"
                    row.extend([str(delta_threshold), str(int(delta_pred)), delta_decision])

                buffer_rows.append(row)

    ap_mean = sum(d_ap_all) / max(1, len(d_ap_all))
    an_mean = sum(d_an_all) / max(1, len(d_an_all))

    if save_csv_path is not None:
        out_dir = os.path.dirname(save_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        header = ["anchor_path", "pos_path", "neg_path", "d_ap", "d_an", "delta"]
        if ap_threshold is not None:
            header.extend(["ap_threshold", "ap_pred", "ap_decision"])
        if an_threshold is not None:
            header.extend(["an_threshold", "an_pred", "an_decision"])
        if delta_threshold is not None:
            header.extend(["delta_threshold", "delta_pred", "delta_decision"])

        with open(save_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(buffer_rows)

    return {"ap_mean": ap_mean, "an_mean": an_mean}