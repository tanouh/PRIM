#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List, Dict
from pathlib import Path

import pandas as pd
import numpy as np


TRIPLET_STANDARD_COLS = ["split", "path_anchor", "path_pos", "path_neg"]


def extract_parcel_id(filename: str) -> str:
    """
    Extract parcel id from filename: id_XX or id_XXX
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


def generate_triplets_from_images(parcel_images: Dict[str, List[str]], min_per_parcel: int = 10, seed: int = 42) -> List[tuple]:
    """
    Generate triplets: at least min_per_parcel triplets for each parcel.
    
    Triplet: (anchor, positive, negative)
    - Anchor and positive: from same parcel
    - Negative: from different parcel
    
    Returns: [(path_anchor, path_pos, path_neg, parcel_id), ...]
    """
    rng = np.random.default_rng(seed)
    triplets = []
    parcel_ids = list(parcel_images.keys())
    
    for parcel_id in parcel_ids:
        images = parcel_images[parcel_id]
        
        if len(images) < 2:
            continue  # Need at least 2 images for anchor + positive
        
        # Collect all other images for negatives
        other_images = []
        for other_id in parcel_ids:
            if other_id != parcel_id:
                other_images.extend(parcel_images[other_id])
        
        if not other_images:
            continue  # Need negatives from other parcels
        
        # Generate min_per_parcel triplets for this parcel
        for _ in range(min_per_parcel):
            # Random anchor and positive from same parcel
            anchor, positive = rng.choice(images, size=2, replace=False)
            # Random negative from different parcel
            negative = rng.choice(other_images)
            triplets.append((anchor, positive, negative, parcel_id))
    
    return triplets


def assign_splits_by_parcel(
    triplets: List[tuple],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign train/validation/test splits at parcel level.
    
    Each triplet has (anchor, positive, negative, parcel_id).
    Split based on parcel_id.
    """
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1, "Invalid split ratios"

    rng = np.random.default_rng(seed)
    
    # Get unique parcels and shuffle
    unique_parcels = list(set(t[3] for t in triplets))
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
    
    # Assign triplets to splits
    data = []
    for anchor, positive, negative, parcel_id in triplets:
        split = parcel_split_map[parcel_id]
        data.append({
            "split": split,
            "path_anchor": anchor,
            "path_pos": positive,
            "path_neg": negative,
        })
    
    return pd.DataFrame(data)


def main():
    ap = argparse.ArgumentParser(
        description="Generate triplet dataset from image folders with parcel-level splitting (cartesian approach)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Approach: 
  1. Generate at least N triplets per parcel (random sampling)
  2. Assign splits at parcel level
  3. Simple and memory-efficient
  
Examples:
  # From single folder
  python scripts/generate_triplets.py --folders data/drive --out csv/triplets.csv
  
  # From multiple folders
  python scripts/generate_triplets.py --folders data/drive data/TAMPAR --out csv/triplets.csv
  
  # Custom splits and minimum training per parcel
  python scripts/generate_triplets.py --folders data/drive --out csv/triplets.csv \
    --train_ratio 0.8 --val_ratio 0.1 --min_train 10 --seed 123
        """
    )
    ap.add_argument("--folders", nargs="+", required=True, help="Image folders to scan")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio (default: 0.7)")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    ap.add_argument("--min_train", type=int, default=10, help="Minimum triplets per parcel (default: 10)")
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

    # Step 1: Generate triplets (at least min_train per parcel)
    print(f"🔄 Generating {args.min_train} triplets per parcel...")
    triplets = generate_triplets_from_images(
        parcel_images,
        min_per_parcel=args.min_train,
        seed=args.seed
    )
    print(f"✅ Generated {len(triplets)} triplets\n")

    # Step 2: Assign splits at parcel level
    print("🔄 Assigning splits (parcel-level)...")
    out_df = assign_splits_by_parcel(
        triplets,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if out_df.empty:
        print("❌ No triplets generated after split assignment.")
        return

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ Triplet generation complete")
    print(f"{'='*70}")
    print(f"Saved {len(out_df)} triplets to {args.out}")
    
    split_counts = out_df["split"].value_counts().to_dict()
    print(f"\nSplit distribution:")
    for split in ["train", "validation", "test"]:
        count = split_counts.get(split, 0)
        print(f"  {split:12s}: {count:6d} triplets")
    
    # Parcel count per split
    print(f"\nParcels per split:")
    for split in ["train", "validation", "test"]:
        df_split = out_df[out_df["split"] == split]
        if len(df_split) > 0:
            n_parcels = len(df_split["path_anchor"].apply(extract_parcel_id).unique())
            print(f"  {split:12s}: {n_parcels:3d} parcels")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()