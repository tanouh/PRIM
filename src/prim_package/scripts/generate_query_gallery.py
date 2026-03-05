#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List, Dict
from pathlib import Path

import pandas as pd
import numpy as np


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


def split_gallery_query(
    parcel_images: Dict[str, List[str]],
    gallery_ratio: float = 0.5,
    min_gallery: int = 1,
    min_query: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Split images per parcel into gallery and query sets.
    
    Gallery: images used to build reference database
    Query: images used to search against gallery
    
    Returns DataFrame with columns: [image_path, label, split]
    """
    rng = np.random.default_rng(seed)
    rows = []
    
    for parcel_id, images in sorted(parcel_images.items()):
        if len(images) < min_gallery + min_query:
            print(f"⚠️  Skipping {parcel_id}: only {len(images)} images (need {min_gallery + min_query})")
            continue
        
        # Shuffle images
        images_array = np.array(images)
        rng.shuffle(images_array)
        
        n = len(images_array)
        n_gallery = max(min_gallery, int(n * gallery_ratio))
        n_gallery = min(n_gallery, n - min_query)  # Ensure min_query remains
        
        gallery_imgs = images_array[:n_gallery]
        query_imgs = images_array[n_gallery:]
        
        # Add to rows
        for img_path in gallery_imgs:
            rows.append({
                "image_path": img_path,
                "label": parcel_id,
                "split": "gallery",
            })
        
        for img_path in query_imgs:
            rows.append({
                "image_path": img_path,
                "label": parcel_id,
                "split": "query",
            })
    
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Generate gallery/query dataset from image folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Approach:
  1. Collect images from folders, grouped by parcel ID
  2. Split each parcel's images into gallery (reference) and query (search) sets
  3. Gallery: used to build the reference database
  4. Query: used to search and retrieve matches from gallery
  
Examples:
  # From single folder
  python scripts/generate_query_gallery.py --folders data/drive --out csv/gallery_query.csv
  
  # From multiple folders
  python scripts/generate_query_gallery.py --folders data/drive data/TAMPAR --out csv/gallery_query.csv
  
  # Custom gallery/query ratio
  python scripts/generate_query_gallery.py --folders data/drive data/TAMPAR \\
    --out csv/gallery_query.csv --gallery_ratio 0.7 --seed 123
        """
    )
    ap.add_argument("--folders", nargs="+", required=True, help="Image folders to scan")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--gallery_ratio", type=float, default=0.5, 
                    help="Ratio of images per parcel to use as gallery (default: 0.5)")
    ap.add_argument("--min_gallery", type=int, default=1, 
                    help="Minimum gallery images per parcel (default: 1)")
    ap.add_argument("--min_query", type=int, default=1, 
                    help="Minimum query images per parcel (default: 1)")
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

    # Split into gallery/query
    print("🔄 Splitting into gallery and query sets...")
    out_df = split_gallery_query(
        parcel_images,
        gallery_ratio=args.gallery_ratio,
        min_gallery=args.min_gallery,
        min_query=args.min_query,
        seed=args.seed,
    )

    if out_df.empty:
        print("❌ No images generated after split.")
        return

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ Gallery/Query generation complete")
    print(f"{'='*70}")
    print(f"Saved {len(out_df)} images to {args.out}")
    
    split_counts = out_df["split"].value_counts().to_dict()
    print(f"\nSplit distribution:")
    print(f"  gallery: {split_counts.get('gallery', 0):6d} images")
    print(f"  query:   {split_counts.get('query', 0):6d} images")
    
    # Parcel counts
    gallery_parcels = len(out_df[out_df["split"] == "gallery"]["label"].unique())
    query_parcels = len(out_df[out_df["split"] == "query"]["label"].unique())
    print(f"\nParcels:")
    print(f"  gallery: {gallery_parcels:3d} parcels")
    print(f"  query:   {query_parcels:3d} parcels")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
