from __future__ import annotations

import os
from typing import Optional, Union, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class PairImageDataset(Dataset):
    """
    Minimal pair dataset for contrastive training.

    Expected DataFrame columns:
      - split: 'train'/'validation' (optional; defaults to 'train' if absent)
      - path_a, path_b, label  (preferred)
        or legacy fallback: a_image_path, b_image_path, label_same

    Behavior:
      - Ignores any bbox/cropping information if present.
      - Loads both images from provided paths and applies the given transform.
      - Returns (img1, img2, y) with y in {0.0, 1.0}.
    """

    def __init__(self, df: pd.DataFrame, root_dir: str = "", transform=None, return_paths: bool = False):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self) -> int:
        return len(self.df)

    def _join(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.root_dir, p)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Determine path columns
        if 'path_a' in row and 'path_b' in row:
            p1 = self._join(str(row['path_a']))
            p2 = self._join(str(row['path_b']))
        elif 'a_image_path' in row and 'b_image_path' in row:
            p1 = self._join(str(row['a_image_path']))
            p2 = self._join(str(row['b_image_path']))
        else:
            raise KeyError("Could not find path columns. Expected ('path_a','path_b') "
                           "or ('a_image_path','b_image_path').")

        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Determine label
        if 'label' in row:
            y_val = float(row['label'])
        elif 'label_same' in row:
            y_val = float(row['label_same'])
        else:
            y_val = 1.0

        y = torch.tensor(y_val, dtype=torch.float32)
        if self.return_paths:
            return img1, img2, y, p1, p2
        return img1, img2, y


class TripletImageDataset(Dataset):
    """
    Triplet dataset for triplet loss training.

    Expected DataFrame columns (preferred):
      - path_anchor, path_pos, path_neg
    Legacy fallbacks:
      - a_image_path, p_image_path, n_image_path
      - anchor_path, positive_path, negative_path

    Returns:
      - (anchor, positive, negative) tensors
    """

    def __init__(self, df: pd.DataFrame, root_dir: str = "", transform=None, return_paths: bool = False):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self) -> int:
        return len(self.df)

    def _join(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.root_dir, p)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Resolve path columns
        def get_val(keys: List[str]) -> str:
            for k in keys:
                if k in row:
                    return str(row[k])
            raise KeyError(f"None of the keys {keys} were found in the row.")

        a_path = self._join(get_val(['path_anchor', 'a_image_path', 'anchor_path']))
        p_path = self._join(get_val(['path_pos', 'p_image_path', 'positive_path']))
        n_path = self._join(get_val(['path_neg', 'n_image_path', 'negative_path']))

        a = Image.open(a_path).convert("RGB")
        p = Image.open(p_path).convert("RGB")
        n = Image.open(n_path).convert("RGB")

        if self.transform:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)

        if self.return_paths:
            return a, p, n, a_path, p_path, n_path
        return a, p, n


def load_pair_dfs(csv_paths: List[str]) -> pd.DataFrame:
    """Load and concatenate CSVs for pair training."""
    dfs = [pd.read_csv(p) for p in csv_paths]
    df_all = pd.concat(dfs, ignore_index=True)
    if 'split' in df_all.columns:
        df_all['split'] = df_all['split'].astype(str).str.lower()
    else:
        df_all['split'] = 'train'
    return df_all


def load_triplet_dfs(csv_paths: List[str]) -> pd.DataFrame:
    """Load and concatenate CSVs for triplet training."""
    dfs = [pd.read_csv(p) for p in csv_paths]
    df_all = pd.concat(dfs, ignore_index=True)
    if 'split' in df_all.columns:
        df_all['split'] = df_all['split'].astype(str).str.lower()
    else:
        df_all['split'] = 'train'
    return df_all


def get_split(df: pd.DataFrame, split_names: Union[str, List[str]]) -> pd.DataFrame:
    """Return subset of df for given split name(s)."""
    if isinstance(split_names, str):
        split_names = [split_names]
    names = [s.lower() for s in split_names]
    return df[df['split'].isin(names)].copy()