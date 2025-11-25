#!/usr/bin/env python3
import os
import re
import csv
import random
from pathlib import Path
from itertools import combinations

SEED = 1337
TRAIN_RATIO = 0.9
DATA_ROOT = Path('data/TAMPAR/test')
OUTPUT_CSV = Path('tampar_pairs_ssl.csv')
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def find_images(root: Path):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in IMG_EXTS:
                fpath = Path(dirpath) / fname
                files.append(fpath)
    return files

def extract_id(path: Path):
    m = re.search(r'id_(\d+)', path.name)
    return int(m.group(1)) if m else None

def group_by_id(paths):
    id_to_paths = {}
    for p in paths:
        idn = extract_id(p)
        if idn is None:
            continue
        id_to_paths.setdefault(idn, []).append(p)
    return id_to_paths

def stable_relpath(p: Path):
    return Path(os.path.relpath(p, Path('.'))).as_posix()

def make_positive_pairs(paths):
    # returns list of (a,b)
    pairs = []
    if len(paths) < 2:
        return pairs
    # ensure deterministic ordering inside id to deduplicate orientation
    spaths = sorted([stable_relpath(p) for p in paths])
    for a, b in combinations(spaths, 2):
        pairs.append((a, b))
    return pairs

def sample_negative_pairs(id_to_paths_subset, num_needed, rng):
    # Build all cross-id combinations
    ids = sorted(id_to_paths_subset.keys())
    pool = []
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            ai = sorted([stable_relpath(p) for p in id_to_paths_subset[ids[i]]])
            bj = sorted([stable_relpath(p) for p in id_to_paths_subset[ids[j]]])
            for a in ai:
                for b in bj:
                    if a < b:
                        pool.append((a,b))
                    else:
                        pool.append((b,a))
    # deduplicate
    pool = list(dict.fromkeys(pool))
    if not pool:
        return []
    if len(pool) <= num_needed:
        return pool  # not enough negatives, return all
    # sample without replacement deterministically
    indices = list(range(len(pool)))
    rng.shuffle(indices)
    chosen = [pool[i] for i in indices[:num_needed]]
    return chosen

def main():
    rng = random.Random(SEED)
    all_images = find_images(DATA_ROOT)
    if not all_images:
        print(f'No images found under {DATA_ROOT}')
        return
    id_to_paths = group_by_id(all_images)
    # remove ids with 0 paths (shouldn't happen)
    id_to_paths = {k:v for k,v in id_to_paths.items() if v}
    all_ids = sorted(id_to_paths.keys())
    if not all_ids:
        print('No valid id_### patterns found.')
        return

    # Split ids 90/10 at object (id) level
    ids_shuffled = all_ids[:]
    rng.shuffle(ids_shuffled)
    n_train = max(1, int(round(len(ids_shuffled) * TRAIN_RATIO)))
    n_train = min(n_train, len(ids_shuffled)-1) if len(ids_shuffled) > 1 else 1
    train_ids = set(ids_shuffled[:n_train])
    val_ids = set(ids_shuffled[n_train:]) if len(ids_shuffled) > 1 else set()

    # Build split-specific dicts
    train_dict = {i: id_to_paths[i] for i in train_ids}
    val_dict = {i: id_to_paths[i] for i in val_ids}

    def build_split_rows(split_name, subset_dict):
        # positives
        pos_pairs = []
        for i, paths in subset_dict.items():
            pos_pairs.extend(make_positive_pairs(paths))
        # negatives
        neg_pairs = sample_negative_pairs(subset_dict, len(pos_pairs), rng)
        rows = []
        for a,b in pos_pairs:
            rows.append((split_name, a, b, 1))
        for a,b in neg_pairs:
            rows.append((split_name, a, b, 0))
        return rows, len(pos_pairs), len(neg_pairs)

    train_rows, n_pos_train, n_neg_train = build_split_rows('train', train_dict)
    val_rows, n_pos_val, n_neg_val = build_split_rows('validation', val_dict)

    rows = train_rows + val_rows

    # write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['split','path_a','path_b','label'])
        for split, a, b, label in rows:
            writer.writerow([split, a, b, label])

    # Print summary
    print('Summary:')
    print(f'  Objects total: {len(all_ids)} | train: {len(train_ids)} | validation: {len(val_ids)}')
    print(f'  Train positives: {n_pos_train} | negatives: {n_neg_train}')
    print(f'  Validation positives: {n_pos_val} | negatives: {n_neg_val}')
    print(f'  CSV written to: {OUTPUT_CSV}')

if __name__ == '__main__':
    main()