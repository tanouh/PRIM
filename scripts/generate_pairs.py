#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd


PAIR_STANDARD_COLS = ["split", "path_a", "path_b", "label"]


def load_and_standardize(csv_path: str, root_dir: str = "") -> pd.DataFrame:
    """
    Standardize input CSV into columns:
      split, path_a, path_b, label

    Supported input column variants:
      - path_a/path_b/label
      - a_image_path/b_image_path/label_same
      - a_image_path/b_image_path/label
    """
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

    # ensure relative root prefix if provided
    if root_dir:
        df["path_a"] = df["path_a"].astype(str).apply(lambda p: p if os.path.isabs(p) else os.path.join(root_dir, p))
        df["path_b"] = df["path_b"].astype(str).apply(lambda p: p if os.path.isabs(p) else os.path.join(root_dir, p))

    # keep only standard cols
    return df[PAIR_STANDARD_COLS].copy()


def main():
    ap = argparse.ArgumentParser("Standardize and merge pair CSVs")
    ap.add_argument("--csv", nargs="+", required=True, help="Input pair CSVs")
    ap.add_argument("--root_dir", default="", help="Optional root to prepend to relative paths")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    dfs: List[pd.DataFrame] = []
    for p in args.csv:
        dfs.append(load_and_standardize(p, root_dir=args.root_dir))

    out_df = pd.concat(dfs, ignore_index=True)
    # normalize split names
    out_df["split"] = out_df["split"].astype(str).str.lower()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()