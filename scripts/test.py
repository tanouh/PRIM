#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from prim_package import (
    PairImageDataset,
    TripletImageDataset,
    load_pair_dfs,
    load_triplet_dfs,
    get_split,
    SiameseNet,
    validate_contrastive,
    validate_triplet,
)
from prim_package.data_processing.transforms import get_eval_transforms
from prim_package.training.losses import pairwise_distance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Siamese model (contrastive or triplet)")
    p.add_argument("--objective", choices=["contrastive", "triplet"], default="contrastive")
    p.add_argument("--csv", nargs="+", required=True, help="One or more CSV paths for evaluation data")
    p.add_argument("--root_dir", default="", help="Optional root prefix to prepend to relative paths in CSV")
    p.add_argument("--model_path", required=True, help="Path to saved model state_dict (from training)")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--distance", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--im_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--pretrained", type=int, default=1, help="1=True, 0=False for encoder pretrained weights")
    # Contrastive output
    p.add_argument(
        "--val_pairs_csv_out",
        default="",
        help="Optional path to save per-pair validation predictions (contrastive only)",
    )
    p.add_argument(
        "--val_threshold",
        type=float,
        default=0.5,
        help="Threshold for decision (distance < threshold => similar). Required for metrics computation.",
    )
    # Triplet output
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


def compute_contrastive_metrics(model, loader, device, distance: str, threshold: float):
    """
    Compute accuracy, precision, recall, F1 for contrastive pairs.
    Prediction: distance < threshold => similar (label=1), else dissimilar (label=0)
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 5:
                img1, img2, y, _, _ = batch
            else:
                img1, img2, y = batch
            
            img1, img2 = img1.to(device), img2.to(device)
            z1, z2 = model(img1, img2)
            d = pairwise_distance(z1, z2, mode=distance).cpu().numpy()
            
            # Predict: 1 if distance < threshold (similar), else 0 (dissimilar)
            pred = (d < threshold).astype(int)
            
            y_true.extend(y.cpu().numpy().astype(int))
            y_pred.extend(pred)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def compute_triplet_metrics(model, loader, device, distance: str):
    """
    Compute accuracy for triplet ordering.
    Correct prediction: d(anchor, positive) < d(anchor, negative)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 6:
                a, p, n, _, _, _ = batch
            else:
                a, p, n = batch
            
            a, p, n = a.to(device), p.to(device), n.to(device)
            
            za = model.forward_once(a)
            zp = model.forward_once(p)
            zn = model.forward_once(n)
            
            d_ap = pairwise_distance(za, zp, mode=distance).cpu().numpy()
            d_an = pairwise_distance(za, zn, mode=distance).cpu().numpy()
            
            # Correct if d(a,p) < d(a,n)
            correct += np.sum(d_ap < d_an)
            total += len(d_ap)
    
    accuracy = correct / max(1, total)
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    im_size = int(args.im_size)
    eval_tf = get_eval_transforms(im_size=im_size)

    # DataFrames
    if args.objective == "contrastive":
        df_all = load_pair_dfs(args.csv)
        test_df = get_split(df_all, ["test", "validation", "valid", "val"]) 
        # prefer explicit 'test' if available
        if len(test_df) == 0:
            test_df = get_split(df_all, "test")

        test_ds = PairImageDataset(
            test_df,
            root_dir=args.root_dir,
            transform=eval_tf,
            return_paths=bool(args.val_pairs_csv_out),
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
        )

    else:
        df_all = load_triplet_dfs(args.csv)
        test_df = get_split(df_all, ["test", "validation", "valid", "val"]) 
        if len(test_df) == 0:
            test_df = get_split(df_all, "test")

        test_ds = TripletImageDataset(
            test_df,
            root_dir=args.root_dir,
            transform=eval_tf,
            return_paths=bool(args.val_triplets_csv_out),
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
        )

    # Load model
    model = SiameseNet(embed_dim=args.embed_dim, pretrained=bool(args.pretrained)).to(device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    state = torch.load(args.model_path, map_location=device, weights_only=True)
    # state might be a full checkpoint or just state_dict
    if isinstance(state, dict) and all(k.startswith("module.") for k in state.keys()):
        # possibly saved from DataParallel; try to load directly
        try:
            model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
        except Exception:
            model.load_state_dict(state)
    else:
        try:
            model.load_state_dict(state)
        except Exception:
            # maybe checkpoint with extra keys
            if "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            else:
                model.load_state_dict(state)

    model.eval()

    # Run evaluation
    if args.objective == "contrastive":
        # Basic validation stats (mean distances)
        val_stats = validate_contrastive(
            model,
            test_loader,
            device,
            distance=args.distance,
            save_csv_path=(args.val_pairs_csv_out if args.val_pairs_csv_out else None),
            threshold=args.val_threshold,
        )
        print("\n" + "="*60)
        print("CONTRASTIVE EVALUATION RESULTS")
        print("="*60)
        print(f"Distance metric: {args.distance}")
        print(f"Threshold: {args.val_threshold}")
        print(f"\nMean distances:")
        print(f"  Positive pairs (similar):     {val_stats.get('pos_mean', float('nan')):.4f}")
        print(f"  Negative pairs (dissimilar):  {val_stats.get('neg_mean', float('nan')):.4f}")
        
        # Compute classification metrics
        metrics = compute_contrastive_metrics(
            model, test_loader, device, args.distance, args.val_threshold
        )
        
        print(f"\nClassification Metrics (threshold={args.val_threshold}):")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-score:  {metrics['f1_score']:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(
            metrics['y_true'], 
            metrics['y_pred'],
            target_names=['Dissimilar (0)', 'Similar (1)'],
            zero_division=0
        ))
        print("="*60 + "\n")

    else:
        # Basic validation stats (mean distances)
        val_stats = validate_triplet(
            model,
            test_loader,
            device,
            distance=args.distance,
            save_csv_path=(args.val_triplets_csv_out if args.val_triplets_csv_out else None),
            ap_threshold=args.val_ap_threshold,
            an_threshold=args.val_an_threshold,
            delta_threshold=args.val_delta_threshold,
        )
        print("\n" + "="*60)
        print("TRIPLET EVALUATION RESULTS")
        print("="*60)
        print(f"Distance metric: {args.distance}")
        print(f"\nMean distances:")
        print(f"  Anchor-Positive (d_ap):  {val_stats.get('ap_mean', float('nan')):.4f}")
        print(f"  Anchor-Negative (d_an):  {val_stats.get('an_mean', float('nan')):.4f}")
        
        # Compute triplet ordering accuracy
        metrics = compute_triplet_metrics(model, test_loader, device, args.distance)
        
        print(f"\nTriplet Ordering Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Correct:  {metrics['correct']} / {metrics['total']}")
        print(f"  (Correct = cases where d(anchor,pos) < d(anchor,neg))")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
