#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
import numpy as np


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

    # split (normalize if provided)
    if "split" in df.columns:
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
        "split": df["split"] if "split" in df.columns else "",
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


def extract_parcel_id(path: str) -> str:
    """
    Extract parcel id from filename: id_XX or id_XXX
    """
    base = os.path.basename(str(path))
    parts = base.split("_")
    if len(parts) >= 2 and parts[0] == "id":
        return f"{parts[0]}_{parts[1]}"
    return ""


def assign_splits(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int = 42) -> pd.DataFrame:
    """
    Assign train/validation/test splits by rows (no stratification for triplets).
    """
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1, "Invalid split ratios"

    rng = np.random.default_rng(seed)
    df = df.copy()

    n = len(df)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    splits = np.array(["test"] * n, dtype=object)
    splits[idx[:n_train]] = "train"
    splits[idx[n_train:n_train + n_val]] = "validation"

    df["split"] = splits
    return df


def assign_splits_by_parcel(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int = 42,
                            allow_cross_split_negative: bool = False) -> pd.DataFrame:
    """
    Parcel-level splitting for triplets:
    - Determine parcel ids for anchor, pos, neg
    - Assign split based on anchor parcel id mapping
    - Drop triplets where pos/neg split differs from anchor (unless allow_cross_split_negative=True for neg)
    """
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1, "Invalid split ratios"

    rng = np.random.default_rng(seed)
    df = df.copy()

    pid_anchor = df["path_anchor"].astype(str).apply(extract_parcel_id)
    pid_pos = df["path_pos"].astype(str).apply(extract_parcel_id)
    pid_neg = df["path_neg"].astype(str).apply(extract_parcel_id)

    df["parcel_id_anchor"] = pid_anchor
    df["parcel_id_pos"] = pid_pos
    df["parcel_id_neg"] = pid_neg

    unique_parcels = sorted(set(pid_anchor.tolist()) | set(pid_pos.tolist()) | set(pid_neg.tolist()))
    parcels = np.array(unique_parcels)
    rng.shuffle(parcels)
    n = len(parcels)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    parcel_split_map = {p: ("train" if i < n_train else ("validation" if i < n_train + n_val else "test"))
                        for i, p in enumerate(parcels)}

    splits = []
    dropped = []
    for i, (pa, pp, pn) in enumerate(zip(df["parcel_id_anchor"], df["parcel_id_pos"], df["parcel_id_neg"])):
        sa = parcel_split_map.get(pa, "")
        sp = parcel_split_map.get(pp, "")
        sn = parcel_split_map.get(pn, "")
        # require anchor/pos same split; neg same split by default
        if sa == sp and (sn == sa or allow_cross_split_negative):
            splits.append(sa)
        else:
            splits.append(None)
            dropped.append(i)

    df["split"] = splits
    if dropped:
        df = df.drop(index=dropped).reset_index(drop=True)

    return df


def main():
    ap = argparse.ArgumentParser("Standardize, merge and split triplet CSVs")
    ap.add_argument("--csv", nargs="+", required=True, help="Input triplet CSVs")
    ap.add_argument("--root_dir", default="", help="Optional root to prepend to relative paths")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--split_mode", choices=["keep", "random", "parcel"], default="parcel",
                    help="Use existing split (keep), row-level random (random), or parcel-level (parcel)")
    ap.add_argument("--train_ratio", type=float, default=0.7, help="Train ratio for random split")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio for random split")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    ap.add_argument("--allow_cross_split_negative", action="store_true",
                    help="For parcel-level split, allow negative to cross split based on anchor")
    args = ap.parse_args()

    dfs: List[pd.DataFrame] = [load_and_standardize(p, root_dir=args.root_dir) for p in args.csv]
    out_df = pd.concat(dfs, ignore_index=True)
    # assign or normalize split
    if args.split_mode == "random":
        out_df = assign_splits(out_df, args.train_ratio, args.val_ratio, seed=args.seed)
    elif args.split_mode == "parcel":
        out_df = assign_splits_by_parcel(
            out_df,
            args.train_ratio,
            args.val_ratio,
            seed=args.seed,
            allow_cross_split_negative=args.allow_cross_split_negative,
        )
    else:
        if "split" not in out_df.columns or (out_df["split"] == "").all():
            out_df["split"] = "train"
        out_df["split"] = out_df["split"].astype(str).str.lower()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    # summary
    split_counts = out_df["split"].value_counts().to_dict()
    print(f"Wrote {len(out_df)} rows to {args.out}")
    print(f"Split distribution: {split_counts}")


if __name__ == "__main__":
    main()