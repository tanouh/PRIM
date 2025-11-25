#!/usr/bin/env python3
import os
import re
import csv
import random
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Optional

SEED = 1337
TRAIN_RATIO = 0.9
DATA_ROOT = Path('data/TAMPAR/test')
OUTPUT_CSV = Path('tampar_triplets_ssl.csv')
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def find_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in IMG_EXTS:
                fpath = Path(dirpath) / fname
                files.append(fpath)
    return files

def extract_id(path: Path) -> Optional[int]:
    m = re.search(r'id_(\d+)', path.name)
    return int(m.group(1)) if m else None

def stable_relpath(p: Path) -> str:
    return Path(os.path.relpath(p, Path('.'))).as_posix()

def group_by_id(paths: List[Path]) -> Dict[int, List[Path]]:
    id_to_paths: Dict[int, List[Path]] = {}
    for p in paths:
        idn = extract_id(p)
        if idn is None:
            continue
        id_to_paths.setdefault(idn, []).append(p)
    # drop ids with fewer than 2 images (cannot make positives or triplets)
    id_to_paths = {k: v for k, v in id_to_paths.items() if len(v) >= 2}
    return id_to_paths

def make_positive_pairs(paths: List[Path]) -> List[Tuple[str, str]]:
    # Deterministic ordering and canonicalized pair orientation to avoid duplicates
    spaths = sorted(stable_relpath(p) for p in paths)
    return list(combinations(spaths, 2))  # (a,b) with a < b

def split_ids(all_ids: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    ids_shuffled = all_ids[:]
    rng.shuffle(ids_shuffled)
    if len(ids_shuffled) == 1:
        # cannot have validation with only 1 id; keep it all in train
        return ids_shuffled, []
    n_train = max(1, int(round(len(ids_shuffled) * TRAIN_RATIO)))
    n_train = min(n_train, len(ids_shuffled) - 1)
    return ids_shuffled[:n_train], ids_shuffled[n_train:]

def build_triplets_for_split(
    subset_dict: Dict[int, List[Path]],
    rng: random.Random,
    split_name: str
) -> Tuple[List[Tuple[str, str, str, str]], int]:
    """
    Returns:
      - rows: list of (split, anchor, same, different)
      - n_pos_pairs: number of positive pairs generated (thus triplets)
    """
    # Prepare canonical path lists per id and a global list per id exclusion
    id_to_canon_paths: Dict[int, List[str]] = {
        i: sorted(stable_relpath(p) for p in plist) for i, plist in subset_dict.items()
    }
    ids = sorted(id_to_canon_paths.keys())

    # Precompute all paths for fast exclusion
    all_paths_list = []
    for i in ids:
        all_paths_list.extend(id_to_canon_paths[i])

    # For each id, compute candidate negatives (all images not of this id)
    id_to_negative_candidates: Dict[int, List[str]] = {}
    for i in ids:
        candidates = [p for p in all_paths_list if p not in id_to_canon_paths[i]]
        id_to_negative_candidates[i] = candidates

    rows: List[Tuple[str, str, str, str]] = []
    n_pos_pairs_total = 0

    for i in ids:
        # Positive pairs within this id
        pos_pairs = make_positive_pairs([Path(p) for p in id_to_canon_paths[i]])
        if not pos_pairs:
            continue
        n_pos_pairs_total += len(pos_pairs)

        neg_candidates = id_to_negative_candidates[i]
        if not neg_candidates:
            # cannot form triplets for this id if no negatives exist in split
            continue

        for a, b in pos_pairs:
            # Deterministically choose a negative per positive pair
            neg = rng.choice(neg_candidates)
            anchor, positive = a, b  # use canonical orientation (a < b)
            rows.append((split_name, anchor, positive, neg))

    return rows, n_pos_pairs_total

def main():
    rng = random.Random(SEED)
    all_images = find_images(DATA_ROOT)
    if not all_images:
        print(f'No images found under {DATA_ROOT}')
        return

    id_to_paths = group_by_id(all_images)
    if not id_to_paths:
        print('No ids with at least 2 images found; cannot form triplets.')
        return

    all_ids = sorted(id_to_paths.keys())
    train_ids, val_ids = split_ids(all_ids, rng)

    train_dict = {i: id_to_paths[i] for i in train_ids}
    val_dict = {i: id_to_paths[i] for i in val_ids}

    train_rows, n_pos_train = build_triplets_for_split(train_dict, rng, 'train')
    val_rows, n_pos_val = build_triplets_for_split(val_dict, rng, 'validation')

    rows = train_rows + val_rows

    # Write CSV: split,anchor,positive,negative
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'anchor', 'same', 'different'])
        for split_name, anchor, positive, negative in rows:
            writer.writerow([split_name, anchor, positive, negative])

    print('Summary:')
    print(f'  Objects total (ids with >=2 imgs): {len(all_ids)} | train: {len(train_ids)} | validation: {len(val_ids)}')
    print(f'  Triplets (train): {len(train_rows)} (from {n_pos_train} positive pairs)')
    print(f'  Triplets (validation): {len(val_rows)} (from {n_pos_val} positive pairs)')
    print(f'  CSV written to: {OUTPUT_CSV}')

if __name__ == '__main__':
    main()