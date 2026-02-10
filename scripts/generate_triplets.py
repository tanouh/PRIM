#!/usr/bin/env python3
# build_triplets.py
from __future__ import annotations
import argparse
import os
import pandas as pd
from typing import Dict

# Get the repo root directory (parent of scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def normalize_path(path: str) -> str:
    """Normalize path to be relative to repo root."""
    path = os.path.normpath(path)
    # If absolute, make it relative to repo root
    if os.path.isabs(path):
        path = os.path.relpath(path, REPO_ROOT)
    return path

def load_ocr_lookup(ocr_path: str) -> Dict[str, str]:
    """Creates a dictionary mapping filepaths to text."""
    if not os.path.exists(ocr_path):
        raise FileNotFoundError(f"OCR Index not found at {ocr_path}. Run extract_ocr.py first.")
        
    print(f"Loading OCR index from {ocr_path}...")
    df = pd.read_csv(ocr_path)
    
    # Create dict: normalized_path -> text
    # We normalize paths to be relative to repo root
    return pd.Series(
        df["ocr_text"].fillna("").values,
        index=df["image_path"].apply(normalize_path)
    ).to_dict()

def process_triplets(triplet_csvs, ocr_lookup, root_dir, out_path):
    all_dfs = []
    
    for csv_path in triplet_csvs:
        print(f"Processing {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # 1. Identify Columns
        def get_col(candidates):
            for c in candidates:
                if c in df.columns: return c
            raise KeyError(f"Missing column from {candidates} in {csv_path}")

        a_col = get_col(["path_anchor", "a_image_path", "anchor_path"])
        p_col = get_col(["path_pos", "p_image_path", "positive_path"])
        n_col = get_col(["path_neg", "n_image_path", "negative_path"])
        
        # 2. Resolve & Normalize Paths (relative to repo root)
        def resolve(p):
            p_str = str(p)
            if root_dir and not os.path.isabs(p_str):
                p_str = os.path.join(root_dir, p_str)
            return normalize_path(p_str)

        # Create temporary columns for lookup
        df['norm_anchor'] = df[a_col].apply(resolve)
        df['norm_pos']    = df[p_col].apply(resolve)
        df['norm_neg']    = df[n_col].apply(resolve)
        
        # 3. Inject Text using the lookup dict
        # Use .map() for speed. fillna("") handles missing images gracefully.
        print("Injecting text data...")
        df['text_anchor'] = df['norm_anchor'].map(ocr_lookup).fillna("")
        df['text_pos']    = df['norm_pos'].map(ocr_lookup).fillna("")
        df['text_neg']    = df['norm_neg'].map(ocr_lookup).fillna("")
        
        # Check hit rate
        missing = (df['text_anchor'] == "").sum()
        if missing > 0:
            print(f"Warning: {missing} anchor images did not have matching OCR data.")

        # 4. Final Formatting
        if "split" not in df.columns:
            df["split"] = "train"
        df["split"] = df["split"].astype(str).str.lower()
        
        # Keep paths relative to repo root for portability
        out_df = pd.DataFrame({
            "split": df["split"],
            "path_anchor": df['norm_anchor'],
            "text_anchor": df['text_anchor'],
            "path_pos": df['norm_pos'],
            "text_pos": df['text_pos'],
            "path_neg": df['norm_neg'],
            "text_neg": df['text_neg']
        })
        
        all_dfs.append(out_df)

    # Merge and Save
    final_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"Success! Saved {len(final_df)} triplets to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="List of raw triplet CSVs")
    parser.add_argument("--ocr_index", required=True, help="Path to images_with_text.csv")
    parser.add_argument("--root_dir", default="", help="Root dir to prepend to paths in raw CSVs")
    parser.add_argument("--out", required=True, help="Final training CSV path")
    args = parser.parse_args()
    
    lookup = load_ocr_lookup(args.ocr_index)
    process_triplets(args.csv, lookup, args.root_dir, args.out)