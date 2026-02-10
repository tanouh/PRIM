#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd


# Get the repo root directory (parent of scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAIR_STANDARD_COLS = ["split", "path_a", "text_a", "path_b", "text_b", "label"]


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


def load_and_standardize(csv_path: str, ocr_lookup: Dict[str, str], root_dir: str = "") -> pd.DataFrame:
    """
    Standardize input CSV into columns:
      split, path_a, text_a, path_b, text_b, label

    Supported input column variants:
      - path_a/path_b/label
      - a_image_path/b_image_path/label_same
      - a_image_path/b_image_path/label
    """
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)

    # split
    if "split" not in df.columns:
        df["split"] = "train"
    df["split"] = df["split"].astype(str).str.lower()

    # path_a
    if "path_a" not in df.columns:
        if "a_image_path" in df.columns:
            df["path_a"] = df["a_image_path"]
        else:
            raise KeyError(f"{csv_path}: missing path_a/a_image_path")

    # path_b
    if "path_b" not in df.columns:
        if "b_image_path" in df.columns:
            df["path_b"] = df["b_image_path"]
        else:
            raise KeyError(f"{csv_path}: missing path_b/b_image_path")

    # label
    if "label" not in df.columns:
        if "label_same" in df.columns:
            df["label"] = df["label_same"]
        else:
            # default to positive if not provided
            df["label"] = 1

    # Resolve & Normalize Paths (relative to repo root)
    def resolve(p):
        p_str = str(p)
        if root_dir and not os.path.isabs(p_str):
            p_str = os.path.join(root_dir, p_str)
        return normalize_path(p_str)

    df["path_a"] = df["path_a"].apply(resolve)
    df["path_b"] = df["path_b"].apply(resolve)

    # Inject OCR text using the lookup dict
    print("Injecting text data...")
    df["text_a"] = df["path_a"].map(ocr_lookup).fillna("")
    df["text_b"] = df["path_b"].map(ocr_lookup).fillna("")

    # Check hit rate
    missing_a = (df["text_a"] == "").sum()
    missing_b = (df["text_b"] == "").sum()
    if missing_a > 0:
        print(f"Warning: {missing_a} path_a images did not have matching OCR data.")
    if missing_b > 0:
        print(f"Warning: {missing_b} path_b images did not have matching OCR data.")

    # keep only standard cols
    return df[PAIR_STANDARD_COLS].copy()


def main():
    ap = argparse.ArgumentParser("Standardize and merge pair CSVs with OCR text")
    ap.add_argument("--csv", nargs="+", required=True, help="Input pair CSVs")
    ap.add_argument("--ocr_index", required=True, help="Path to OCR index CSV (e.g., csv/ocr_texts.csv)")
    ap.add_argument("--root_dir", default="", help="Optional root to prepend to relative paths")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    # Load OCR lookup
    ocr_lookup = load_ocr_lookup(args.ocr_index)

    dfs: List[pd.DataFrame] = []
    for p in args.csv:
        dfs.append(load_and_standardize(p, ocr_lookup, root_dir=args.root_dir))

    out_df = pd.concat(dfs, ignore_index=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Success! Saved {len(out_df)} pairs to {args.out}")


if __name__ == "__main__":
    main()