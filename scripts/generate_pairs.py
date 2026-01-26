#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List, Dict
from pathlib import Path
from itertools import combinations, product

import pandas as pd
import numpy as np


PAIR_STANDARD_COLS = ["split", "path_a", "path_b", "label"]


def extract_parcel_id(filename: str) -> str:
    """
    Extract parcel id from filename: id_XX or id_XXX
    Works with naming like id_100_... or id_00_...
    """
    base = os.path.basename(str(filename))
    parts = base.split("_")
    if len(parts) >= 2 and parts[0] == "id":
        return f"{parts[0]}_{parts[1]}"
    return ""


def collect_images_from_folders(folders: List[str]) -> Dict[str, List[str]]:
    """
    Scan folders for images and group by parcel_id.
    Returns: {parcel_id: [path1, path2, ...]}
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    parcel_images = {}
    
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {folder}")
            continue
        
        for img_file in folder_path.glob("*"):
            if img_file.is_file() and img_file.suffix.lower() in valid_exts:
                parcel_id = extract_parcel_id(img_file.name)
                if parcel_id:
                    if parcel_id not in parcel_images:
                        parcel_images[parcel_id] = []
                    parcel_images[parcel_id].append(str(img_file))
    
    return parcel_images


def generate_pairs_from_images(parcel_images: Dict[str, List[str]], min_per_parcel: int = 10, seed: int = 42) -> List[tuple]:
    """
    Generate pairs: at least min_per_parcel pairs for each parcel (mix of positive and negative).
    
    Positive pairs: from same parcel (label=1)
    Negative pairs: from different parcels (label=0)
    
    Returns: [(path_a, path_b, label, parcel_id), ...]
    """
    rng = np.random.default_rng(seed)
    pairs = []
    parcel_ids = list(parcel_images.keys())
    
    for parcel_id in parcel_ids:
        images = parcel_images[parcel_id]
        
        # Collect other images for negatives
        other_images = []
        for other_id in parcel_ids:
            if other_id != parcel_id:
                other_images.extend(parcel_images[other_id])
        
        # Generate pairs: half positive, half negative (approximately)
        n_positive = min_per_parcel // 2
        n_negative = min_per_parcel - n_positive
        
        # Positive pairs (same parcel)
        if len(images) >= 2:
            for _ in range(n_positive):
                img_a, img_b = rng.choice(images, size=2, replace=False)
                pairs.append((img_a, img_b, 1, parcel_id))
        
        # Negative pairs (different parcels)
        if other_images and len(images) > 0:
            for _ in range(n_negative):
                img_a = rng.choice(images)
                img_b = rng.choice(other_images)
                pairs.append((img_a, img_b, 0, parcel_id))
    
    return pairs


def assign_splits_by_parcel(
    pairs: List[tuple],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign train/validation/test splits at parcel level.
    
    Each pair has (path_a, path_b, label, parcel_id).
    Split based on parcel_id.
    """
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1, "Invalid split ratios"

    rng = np.random.default_rng(seed)
    
    # Get unique parcels and shuffle
    unique_parcels = list(set(p[3] for p in pairs))
    parcels = np.array(unique_parcels)
    rng.shuffle(parcels)
    n = len(parcels)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    
    # Create parcel-to-split mapping
    parcel_split_map = {}
    for i, p in enumerate(parcels):
        if i < n_train:
            parcel_split_map[p] = "train"
        elif i < n_train + n_val:
            parcel_split_map[p] = "validation"
        else:
            parcel_split_map[p] = "test"
    
    # Assign pairs to splits
    data = []
    for path_a, path_b, label, parcel_id in pairs:
        split = parcel_split_map[parcel_id]
        data.append({
            "split": split,
            "path_a": path_a,
            "path_b": path_b,
            "label": label,
        })
    
    return pd.DataFrame(data)


def main():
    ap = argparse.ArgumentParser(
        description="Generate pair dataset from image folders with parcel-level splitting (cartesian approach)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Approach:
  1. Generate at least N pairs per parcel (random sampling, mix of positive/negative)
  2. Assign splits at parcel level
  3. Simple and memory-efficient
  
Examples:
  # From single folder
  python scripts/generate_pairs.py --folders data/drive --out csv/pairs.csv
  
  # From multiple folders
  python scripts/generate_pairs.py --folders data/drive data/TAMPAR --out csv/pairs.csv
  
  # Custom splits and minimum training
  python scripts/generate_pairs.py --folders data/drive --out csv/pairs.csv \
    --train_ratio 0.8 --val_ratio 0.1 --min_train 10 --seed 123
        """
    )
    ap.add_argument("--folders", nargs="+", required=True, help="Image folders to scan")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio (default: 0.7)")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    ap.add_argument("--min_train", type=int, default=10, help="Minimum pairs per parcel (default: 10)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = ap.parse_args()

    # Collect images from folders
    print("📂 Scanning folders for images...")
    parcel_images = collect_images_from_folders(args.folders)
    if not parcel_images:
        print("❌ No images found in provided folders.")
        return
    
    print(f"✅ Found {len(parcel_images)} unique parcel IDs")
    total_images = sum(len(imgs) for imgs in parcel_images.values())
    print(f"   {total_images} total images\n")

    # Step 1: Generate pairs (at least min_train per parcel)
    print(f"🔄 Generating {args.min_train} pairs per parcel...")
    pairs = generate_pairs_from_images(
        parcel_images,
        min_per_parcel=args.min_train,
        seed=args.seed
    )
    print(f"✅ Generated {len(pairs)} pairs")
    
    # Count positive/negative
    labels = [label for _, _, label, _ in pairs]
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    print(f"   Positive (label=1): {n_pos}")
    print(f"   Negative (label=0): {n_neg}\n")
    
    # Step 2: Assign splits at parcel level
    print("🔄 Assigning splits (parcel-level)...")
    out_df = assign_splits_by_parcel(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if out_df.empty:
        print("❌ No pairs generated after split assignment.")
        return

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ Pair generation complete")
    print(f"{'='*70}")
    print(f"Saved {len(out_df)} pairs to {args.out}")
    
    split_counts = out_df["split"].value_counts().to_dict()
    print(f"\nSplit distribution:")
    for split in ["train", "validation", "test"]:
        count = split_counts.get(split, 0)
        print(f"  {split:12s}: {count:6d} pairs")
    
    # Label distribution
    label_counts = out_df["label"].value_counts().to_dict()
    print(f"\nLabel distribution:")
    print(f"  positive (1): {label_counts.get(1, 0):6d} pairs")
    print(f"  negative (0): {label_counts.get(0, 0):6d} pairs")
    
    # Parcel count per split
    print(f"\nParcels per split:")
    for split in ["train", "validation", "test"]:
        df_split = out_df[out_df["split"] == split]
        if len(df_split) > 0:
            n_parcels = len(df_split["path_a"].apply(extract_parcel_id).unique())
            print(f"  {split:12s}: {n_parcels:3d} parcels")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()