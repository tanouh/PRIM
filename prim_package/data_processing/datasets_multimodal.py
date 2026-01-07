from __future__ import annotations

import os
from typing import Optional, Union, List, Tuple, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer


class PairImageDatasetMultimodal(Dataset):
    """
    Multimodal pair dataset for contrastive training (Image + Text).

    Expected DataFrame columns:
      - path_a, path_b (images)
      - text_a, text_b (OCR text content) - defaults to "" if missing
      - label (0.0 or 1.0)
    """

    def __init__(
        self, 
        df: pd.DataFrame, 
        root_dir: str = "", 
        transform=None, 
        return_paths: bool = False,
        tokenizer_name: str = "distilbert-base-uncased",
        max_seq_length: int = 64
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.return_paths = return_paths
        
        # Initialize Tokenizer (suppress fast tokenizer warning if needed)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.df)

    def _join(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.root_dir, p)

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenizes text and returns dict with 'input_ids' and 'attention_mask'.
        Squeezes the batch dimension (1, seq_len) -> (seq_len).
        """
        enc = self.tokenizer(
            str(text),
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        # Squeeze because DataLoader will add the batch dimension later
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # --- 1. Load Images ---
        # Determine path columns
        if 'path_a' in row and 'path_b' in row:
            p1 = self._join(str(row['path_a']))
            p2 = self._join(str(row['path_b']))
        elif 'a_image_path' in row and 'b_image_path' in row:
            p1 = self._join(str(row['a_image_path']))
            p2 = self._join(str(row['b_image_path']))
        else:
            raise KeyError("Could not find path columns (path_a/path_b).")

        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # --- 2. Load & Tokenize Text ---
        # Fallback to empty string if column is missing or value is NaN
        t1_str = str(row.get('text_a', row.get('text_content_a', '')))
        t2_str = str(row.get('text_b', row.get('text_content_b', '')))
        
        if t1_str == 'nan': t1_str = ""
        if t2_str == 'nan': t2_str = ""

        txt1 = self._tokenize(t1_str)
        txt2 = self._tokenize(t2_str)

        # --- 3. Label ---
        if 'label' in row:
            y_val = float(row['label'])
        elif 'label_same' in row:
            y_val = float(row['label_same'])
        else:
            y_val = 1.0
        y = torch.tensor(y_val, dtype=torch.float32)

        if self.return_paths:
            return img1, txt1, img2, txt2, y, p1, p2
        
        return img1, txt1, img2, txt2, y


class TripletImageDatasetMultimodal(Dataset):
    """
    Multimodal triplet dataset (Anchor, Positive, Negative).

    Expected DataFrame columns:
      - path_anchor, path_pos, path_neg
      - text_anchor, text_pos, text_neg (optional)
    """

    def __init__(
        self, 
        df: pd.DataFrame, 
        root_dir: str = "", 
        transform=None, 
        return_paths: bool = False,
        tokenizer_name: str = "distilbert-base-uncased",
        max_seq_length: int = 64
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.return_paths = return_paths
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.df)

    def _join(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.root_dir, p)

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            str(text),
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # --- 1. Images ---
        def get_val(keys: List[str], default="") -> str:
            for k in keys:
                if k in row:
                    val = row[k]
                    return "" if pd.isna(val) else str(val)
            return default

        # Resolve paths
        a_path = self._join(get_val(['path_anchor', 'a_image_path', 'anchor_path']))
        p_path = self._join(get_val(['path_pos', 'p_image_path', 'positive_path']))
        n_path = self._join(get_val(['path_neg', 'n_image_path', 'negative_path']))

        a_img = Image.open(a_path).convert("RGB")
        p_img = Image.open(p_path).convert("RGB")
        n_img = Image.open(n_path).convert("RGB")

        if self.transform:
            a_img = self.transform(a_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)

        # --- 2. Text ---
        # Try to find text columns, default to empty string
        txt_a_str = get_val(['text_anchor', 'text_a', 'anchor_text'])
        txt_p_str = get_val(['text_pos', 'text_p', 'pos_text'])
        txt_n_str = get_val(['text_neg', 'text_n', 'neg_text'])

        a_txt = self._tokenize(txt_a_str)
        p_txt = self._tokenize(txt_p_str)
        n_txt = self._tokenize(txt_n_str)

        if self.return_paths:
            return a_img, a_txt, p_img, p_txt, n_img, n_txt, a_path, p_path, n_path
        
        return a_img, a_txt, p_img, p_txt, n_img, n_txt

# --- Helper functions remain mostly unchanged ---

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