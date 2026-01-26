#!/usr/bin/env python3
import os
import re
import csv
import random
import argparse
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Optional

SEED = 1337
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

def extract_id(path: Path) -> Optional[str]:
    """Extract id from folder name (e.g., id_100, id_501)"""
    # Look for id in parent folder name
    for part in path.parts:
        if part.startswith('id_'):
            return part
    return None

def stable_relpath(p: Path) -> str:
    return Path(os.path.relpath(p, Path('.'))).as_posix()

def group_by_id(paths: List[Path]) -> Dict[str, List[Path]]:
    id_to_paths: Dict[str, List[Path]] = {}
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

def build_triplets(
    id_to_paths: Dict[str, List[Path]],
    rng: random.Random,
    split_name: str
) -> Tuple[List[Tuple[str, str, str, str]], int]:
    """
    Returns:
      - rows: list of (split, anchor, same, different)
      - n_pos_pairs: number of positive pairs generated (thus triplets)
    """
    # Prepare canonical path lists per id and a global list per id exclusion
    id_to_canon_paths: Dict[str, List[str]] = {
        i: sorted(stable_relpath(p) for p in plist) for i, plist in id_to_paths.items()
    }
    ids = sorted(id_to_canon_paths.keys())

    # Precompute all paths for fast exclusion
    all_paths_list = []
    for i in ids:
        all_paths_list.extend(id_to_canon_paths[i])

    # For each id, compute candidate negatives (all images not of this id)
    id_to_negative_candidates: Dict[str, List[str]] = {}
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
            # cannot form triplets for this id if no negatives exist
            continue

        for a, b in pos_pairs:
            # Deterministically choose a negative per positive pair
            neg = rng.choice(neg_candidates)
            anchor, positive = a, b  # use canonical orientation (a < b)
            rows.append((split_name, anchor, positive, neg))

    return rows, n_pos_pairs_total

def main():
    parser = argparse.ArgumentParser(description='Generate triplets CSV for test from drive folder')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to data folder (e.g., data/drive)')
    parser.add_argument('--output', type=str, default='csv/drive_triplets_test.csv', help='Output CSV path')
    parser.add_argument('--split', type=str, default='test', help='Split name (default: test)')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_root = Path(args.data_folder)
    
    if not data_root.exists():
        print(f'Error: Data folder {data_root} does not exist')
        return

    all_images = find_images(data_root)
    if not all_images:
        print(f'No images found under {data_root}')
        return

    id_to_paths = group_by_id(all_images)
    if not id_to_paths:
        print('No ids with at least 2 images found; cannot form triplets.')
        return

    all_ids = sorted(id_to_paths.keys())
    
    rows, n_pos_pairs = build_triplets(id_to_paths, rng, args.split)

    # Write CSV: split,anchor,positive,negative
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'path_anchor', 'positive_path', 'negative_path'])
        for split_name, anchor, positive, negative in rows:
            writer.writerow([split_name, anchor, positive, negative])

    print('Summary:')
    print(f'  Data folder: {data_root}')
    print(f'  Objects total (ids with >=2 imgs): {len(all_ids)}')
    print(f'  Triplets ({args.split}): {len(rows)} (from {n_pos_pairs} positive pairs)')
    print(f'  CSV written to: {output_path}')

if __name__ == '__main__':
    main()
