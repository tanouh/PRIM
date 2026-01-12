#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Optional
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from prim_package import (
    SingleImageDataset,
    load_single_df,
    get_split,
    SiameseNet,
)
from prim_package.data_processing.transforms import get_eval_transforms
from prim_package.training.losses import pairwise_distance


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("One-to-many Siamese evaluation")
    p.add_argument("--csv", required=True, help="CSV with image_path,label,split")
    p.add_argument("--root_dir", default="")
    p.add_argument("--model_path", required=True)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--distance", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--im_size", type=int, default=256)
    p.add_argument("--out", type=str, required=True, help="Output directory for results")
    p.add_argument("--threshold", type=float, default=None, help="Optional verification threshold")
    
    return p.parse_args()


# ---------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, loader, device):
    embeddings = []
    labels = []
    paths = []

    for imgs, lbls, pths in loader:
        imgs = imgs.to(device)
        z = model.forward_once(imgs)
        embeddings.append(z.cpu())
        labels.extend(lbls)
        paths.extend(pths)

    return torch.cat(embeddings), np.array(labels), paths


# ---------------------------------------------------------
# One-to-many evaluation
# ---------------------------------------------------------
def evaluate_one_to_many(
    query_embs,
    query_labels,
    gallery_embs,
    gallery_labels,
    distance="cosine"
):
    correct = 0

    for i in range(len(query_embs)):
        q = query_embs[i].unsqueeze(0).expand_as(gallery_embs)
        dists = pairwise_distance(q, gallery_embs, mode=distance)

        nn_idx = torch.argmin(dists).item()
        pred_label = gallery_labels[nn_idx]

        if pred_label == query_labels[i]:
            correct += 1

    return correct / len(query_embs)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = get_eval_transforms(im_size=args.im_size)

    # Load dataframe
    df = load_single_df(args.csv)

    gallery_df = get_split(df, "gallery")
    query_df = get_split(df, "query")

    gallery_ds = SingleImageDataset(
        gallery_df,
        root_dir=args.root_dir,
        transform=tf,
        return_paths=True,
    )

    query_ds = SingleImageDataset(
        query_df,
        root_dir=args.root_dir,
        transform=tf,
        return_paths=True,
    )

    gallery_loader = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False)
    query_loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = SiameseNet(embed_dim=args.embed_dim, pretrained=False).to(device)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Build embeddings
    gallery_embs, gallery_labels, gallery_paths = extract_embeddings(
        model, gallery_loader, device
    )

    query_embs, query_labels, query_paths = extract_embeddings(
        model, query_loader, device
    )

    # One-to-many accuracy (Nearest Neighbor)
    acc = evaluate_one_to_many(
        query_embs,
        query_labels,
        gallery_embs,
        gallery_labels,
        distance=args.distance,
    )

    print("\n" + "=" * 60)
    print("ONE-TO-MANY EVALUATION")
    print("=" * 60)
    print(f"Distance metric: {args.distance}")
    print(f"Top-1 Accuracy: {acc:.4f}")

    # Optional threshold-based verification
    if args.threshold is not None:
        y_true, y_pred = [], []

        for i in range(len(query_embs)):
            q = query_embs[i].unsqueeze(0).expand_as(gallery_embs)
            dists = pairwise_distance(q, gallery_embs, mode=args.distance)
            min_dist = dists.min().item()

            y_true.append(1)  # assume each query has a match
            y_pred.append(1 if min_dist < args.threshold else 0)

        print(f"Verification Accuracy (threshold={args.threshold}): "
              f"{accuracy_score(y_true, y_pred):.4f}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
