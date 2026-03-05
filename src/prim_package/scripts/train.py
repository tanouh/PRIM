#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from prim_package import (
    PairImageDataset,
    TripletImageDataset,
    load_pair_dfs,
    load_triplet_dfs,
    get_split,
    SiameseNet,
    train_contrastive,
    validate_contrastive,
    train_triplet,
    validate_triplet,
)
from prim_package.data_processing.transforms import get_train_transforms, get_eval_transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Siamese with contrastive or triplet loss")
    p.add_argument("--objective", choices=["contrastive", "triplet"], default="contrastive")
    p.add_argument("--csv", nargs="+", required=True, help="One or more CSV paths")
    p.add_argument("--root_dir", default="", help="Optional root prefix to prepend to relative paths in CSV")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--pretrained", type=int, default=1, help="1=True, 0=False for encoder pretrained weights")
    p.add_argument("--distance", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--margin", type=float, default=1.0, help="Margin for loss (0.3 typical for triplet)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--im_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--save_path", default="siamese.pt", help="Where to save the trained model")
    p.add_argument("--sbatch", type=int, default=0, help="1=True, 0=False for sbatch configuration")
    # Validation logging (contrastive only)
    p.add_argument(
        "--val_pairs_csv_out",
        default="",
        help="Optional path to save per-pair validation predictions (contrastive only)",
    )
    p.add_argument(
        "--val_threshold",
        type=float,
        default=None,
        help="Optional threshold for decision (distance < threshold => similar)",
    )
    # Validation logging (triplet only)
    p.add_argument(
        "--val_triplets_csv_out",
        default="",
        help="Optional path to save per-triplet validation logs (triplet only)",
    )
    p.add_argument(
        "--val_ap_threshold",
        type=float,
        default=None,
        help="Optional threshold: d(a,p) < val_ap_threshold => anchor~pos similar",
    )
    p.add_argument(
        "--val_an_threshold",
        type=float,
        default=None,
        help="Optional threshold: d(a,n) > val_an_threshold => anchor~neg dissimilar",
    )
    p.add_argument(
        "--val_delta_threshold",
        type=float,
        default=None,
        help="Optional threshold on delta=d(a,n)-d(a,p): delta > val_delta_threshold => correct ordering",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if bool(args.sbatch):
        args.num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", args.num_workers))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    im_size = int(args.im_size)
    train_tf = get_train_transforms(im_size=im_size)
    eval_tf = get_eval_transforms(im_size=im_size)

    # DataFrames
    if args.objective == "contrastive":
        df_all = load_pair_dfs(args.csv)
        train_df = get_split(df_all, "train")
        val_df = get_split(df_all, ["validation", "valid", "val"])
        if len(val_df) == 0:
            val_df = get_split(df_all, "validation")

        train_ds = PairImageDataset(train_df, root_dir=args.root_dir, transform=train_tf)
        val_ds = PairImageDataset(
            val_df,
            root_dir=args.root_dir,
            transform=eval_tf,
            return_paths=bool(args.val_pairs_csv_out),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
        )

    else:
        df_all = load_triplet_dfs(args.csv)
        train_df = get_split(df_all, "train")
        val_df = get_split(df_all, ["validation", "valid", "val"])
        if len(val_df) == 0:
            val_df = get_split(df_all, "validation")

        train_ds = TripletImageDataset(train_df, root_dir=args.root_dir, transform=train_tf)
        val_ds = TripletImageDataset(
            val_df,
            root_dir=args.root_dir,
            transform=eval_tf,
            return_paths=bool(args.val_triplets_csv_out),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
        )

    # Model and optimizer
    model = SiameseNet(embed_dim=args.embed_dim, pretrained=bool(args.pretrained)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    print(f"Objective: {args.objective}")
    for ep in range(args.epochs):
        if args.objective == "contrastive":
            train_loss = train_contrastive(
                model, train_loader, optimizer, device, margin=args.margin, distance=args.distance
            )
            val_stats = validate_contrastive(
                model,
                val_loader,
                device,
                distance=args.distance,
                save_csv_path=(args.val_pairs_csv_out if args.val_pairs_csv_out else None),
                threshold=args.val_threshold,
            )
            print(
                f"Epoch {ep+1}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val pos_mean={val_stats['pos_mean']:.3f} | "
                f"val neg_mean={val_stats['neg_mean']:.3f}"
            )
        else:
            # Triplet
            train_loss = train_triplet(
                model, train_loader, optimizer, device, margin=args.margin, distance=args.distance
            )
            val_stats = validate_triplet(
                model,
                val_loader,
                device,
                distance=args.distance,
                save_csv_path=(args.val_triplets_csv_out if args.val_triplets_csv_out else None),
                ap_threshold=args.val_ap_threshold,
                an_threshold=args.val_an_threshold,
                delta_threshold=args.val_delta_threshold,
            )
            print(
                f"Epoch {ep+1}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val ap_mean={val_stats['ap_mean']:.3f} | "
                f"val an_mean={val_stats['an_mean']:.3f}"
            )

    # Save model
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    main()