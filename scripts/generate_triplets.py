#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd


TRIPLET_STANDARD_COLS = ["split", "path_anchor", "path_pos", "path_neg"]


def load_and_standardize(csv_path: str, root_dir: str = "") -> pd.DataFrame:
    """
    Standardize input CSV into columns:
      split, path_anchor, path_pos, path_neg

    Supported input column variants (any of the following present per role):
      - anchor:  path_anchor | a_image_path | anchor_path
      - positive: path_pos | p_image_path | positive_path
      - negative: path_neg | n_image_path | negative_path
    """
    df = pd.read_csv(csv_path)

    # split
    if "split" not in df.columns:
        df["split"] = "train"
    df["split"] = df["split"].astype(str).str.lower()

    # helper
    def pick(colnames: List[str]) -> str:
        for c in colnames:
            if c in df.columns:
                return c
        raise KeyError(f"{csv_path}: none of the columns {colnames} found")

    a_col = pick(["path_anchor", "a_image_path", "anchor_path"])
    p_col = pick(["path_pos", "p_image_path", "positive_path"])
    n_col = pick(["path_neg", "n_image_path", "negative_path"])

    out = pd.DataFrame({
        "split": df["split"],
        "path_anchor": df[a_col].astype(str),
        "path_pos": df[p_col].astype(str),
        "path_neg": df[n_col].astype(str),
    })

    # ensure root prefix if provided and paths are not absolute
    if root_dir:
        def join_path(p: str) -> str:
            return p if os.path.isabs(p) else os.path.join(root_dir, p)
        out["path_anchor"] = out["path_anchor"].apply(join_path)
        out["path_pos"] = out["path_pos"].apply(join_path)
        out["path_neg"] = out["path_neg"].apply(join_path)

    return out[TRIPLET_STANDARD_COLS].copy()


def main():
    ap = argparse.ArgumentParser("Standardize and merge triplet CSVs")
    ap.add_argument("--csv", nargs="+", required=True, help="Input triplet CSVs")
    ap.add_argument("--root_dir", default="", help="Optional root to prepend to relative paths")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    dfs: List[pd.DataFrame] = [load_and_standardize(p, root_dir=args.root_dir) for p in args.csv]
    out_df = pd.concat(dfs, ignore_index=True)
    out_df["split"] = out_df["split"].astype(str).str.lower()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()