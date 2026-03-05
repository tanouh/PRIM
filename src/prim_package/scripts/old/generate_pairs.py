#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
import numpy as np


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
    if "split" in df.columns:
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


def extract_parcel_id(path: str) -> str:
    """
    Extract parcel id from filename: id_XX or id_XXX
    Works with flattened naming like id_100_... or id_00_...
    """
    base = os.path.basename(str(path))
    parts = base.split("_")
    if len(parts) >= 2 and parts[0] == "id":
        return f"{parts[0]}_{parts[1]}"
    return ""


def assign_splits(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
    stratify: bool = True,
) -> pd.DataFrame:
    """
    Assign train/validation/test splits by rows.

    If stratify=True and 'label' exists, preserves class balance across splits.
    """
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1, "Invalid split ratios"

    rng = np.random.default_rng(seed)
    df = df.copy()

    def split_indices(n: int) -> np.ndarray:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        # ensure total does not exceed n
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        splits = np.array(["test"] * n, dtype=object)
        splits[idx[:n_train]] = "train"
        splits[idx[n_train:n_train + n_val]] = "validation"
        return splits

    if stratify and "label" in df.columns:
        split_col = pd.Series(index=df.index, dtype=object)
        for label_val, group in df.groupby("label"):
            group_splits = split_indices(len(group))
            split_col.loc[group.index] = group_splits
        df["split"] = split_col.values
    else:
        df["split"] = split_indices(len(df))

    return df


def assign_splits_by_parcel(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
    allow_cross_split_negatives: bool = False,
) -> pd.DataFrame:
    """
    Parcel-level splitting:
    - Build unique parcel ids from both path_a and path_b
    - Randomly assign parcels to train/validation/test
    - For each row:
        - If both parcels map to the same split, use it
        - Else (cross-split):
            - If allow_cross_split_negatives=True and label==0: assign by path_a's parcel split
            - Otherwise: mark for drop
    """
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1, "Invalid split ratios"

    rng = np.random.default_rng(seed)
    df = df.copy()

    # Extract parcel ids
    pid_a = df["path_a"].astype(str).apply(extract_parcel_id)
    pid_b = df["path_b"].astype(str).apply(extract_parcel_id)
    df["parcel_id_a"] = pid_a
    df["parcel_id_b"] = pid_b

    unique_parcels = sorted(set(pid_a.tolist()) | set(pid_b.tolist()))
    parcels = np.array(unique_parcels)
    rng.shuffle(parcels)
    n = len(parcels)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    parcel_split_map = {p: ("train" if i < n_train else ("validation" if i < n_train + n_val else "test"))
                        for i, p in enumerate(parcels)}

    # Assign row splits
    splits = []
    dropped_idx = []
    for i, (a, b, lbl) in enumerate(zip(df["parcel_id_a"], df["parcel_id_b"], df["label"])):
        sa = parcel_split_map.get(a, "")
        sb = parcel_split_map.get(b, "")
        if sa == sb:
            splits.append(sa)
        else:
            if allow_cross_split_negatives and int(lbl) == 0:
                splits.append(sa)
            else:
                splits.append(None)
                dropped_idx.append(i)

    df["split"] = splits
    # Drop rows with None split (cross-split pairs)
    if dropped_idx:
        df = df.drop(index=dropped_idx).reset_index(drop=True)

    return df


def main():
    ap = argparse.ArgumentParser("Standardize, merge and split pair CSVs")
    ap.add_argument("--csv", nargs="+", required=True, help="Input pair CSVs")
    ap.add_argument("--root_dir", default="", help="Optional root to prepend to relative paths")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--split_mode", choices=["keep", "random", "parcel"], default="parcel",
                    help="Use existing split (keep), row-level random (random), or parcel-level (parcel)")
    ap.add_argument("--train_ratio", type=float, default=0.7, help="Train ratio for random split")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio for random split")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    ap.add_argument("--no_stratify", action="store_true", help="Do not stratify by label during row-level split")
    ap.add_argument("--allow_cross_split_negatives", action="store_true",
                    help="For parcel-level split, allow label=0 pairs to cross splits by assigning based on path_a")
    args = ap.parse_args()

    dfs: List[pd.DataFrame] = []
    for p in args.csv:
        dfs.append(load_and_standardize(p, root_dir=args.root_dir))

    out_df = pd.concat(dfs, ignore_index=True)
    # assign or normalize split
    if args.split_mode == "random":
        out_df = assign_splits(
            out_df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            stratify=(not args.no_stratify),
        )
    elif args.split_mode == "parcel":
        out_df = assign_splits_by_parcel(
            out_df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            allow_cross_split_negatives=args.allow_cross_split_negatives,
        )
    else:
        if "split" not in out_df.columns:
            # default all to train if keeping split but none provided
            out_df["split"] = "train"
        out_df["split"] = out_df["split"].astype(str).str.lower()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    # summary
    split_counts = out_df["split"].value_counts().to_dict()
    label_counts = out_df["label"].value_counts().to_dict() if "label" in out_df.columns else {}
    print(f"Wrote {len(out_df)} rows to {args.out}")
    print(f"Split distribution: {split_counts}")
    if label_counts:
        print(f"Label distribution: {label_counts}")


if __name__ == "__main__":
    main()