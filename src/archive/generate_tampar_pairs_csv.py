import argparse
import csv
import random
import re
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Set

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

# Matches 'id_123', 'ID-456', 'id123' anywhere in filename (case-insensitive)
ID_REGEX = re.compile(r'(?i)id[_\-]?(\d+)')

def find_images(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    files: List[Path] = []
    for p in root_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    files.sort()
    return files

def extract_id(p: Path) -> Optional[str]:
    m = ID_REGEX.search(p.stem)
    if m:
        return m.group(1)
    return None

def index_by_id(files: List[Path], root_dir: Path) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    skipped = 0
    for p in files:
        idv = extract_id(p)
        if idv is None:
            skipped += 1
            continue
        try:
            rel = p.relative_to(root_dir).as_posix()
        except ValueError:
            rel = p.as_posix()
        groups[idv].append(rel)
    # sort each group's paths for deterministic ordering
    for k in groups:
        groups[k].sort()
    if skipped > 0:
        print(f'Warning: skipped {skipped} files without an id_* pattern')
    return groups

def build_positive_pairs(groups: Dict[str, List[str]]) -> List[Tuple[str, str, int]]:
    pos: List[Tuple[str, str, int]] = []
    for idv, paths in groups.items():
        if len(paths) < 2:
            continue
        for a, b in combinations(paths, 2):
            a_, b_ = sorted((a, b))
            pos.append((a_, b_, 1))
    return pos

def build_negative_pairs(groups: Dict[str, List[str]], target_neg: int, rng: random.Random) -> List[Tuple[str, str, int]]:
    ids = [k for k, v in groups.items() if len(v) >= 1]
    if len(ids) < 2 or target_neg <= 0:
        return []
    neg_set: Set[Tuple[str, str]] = set()
    neg: List[Tuple[str, str, int]] = []
    # Precompute for speed
    id_to_paths = groups
    max_trials = max(1000, target_neg * 20)
    trials = 0
    while len(neg) < target_neg and trials < max_trials:
        trials += 1
        id_a, id_b = rng.sample(ids, 2)
        pa = rng.choice(id_to_paths[id_a])
        pb = rng.choice(id_to_paths[id_b])
        a, b = sorted((pa, pb))
        key = (a, b)
        if key in neg_set:
            continue
        neg_set.add(key)
        neg.append((a, b, 0))
    return neg

def write_pairs_csv(pairs: List[Tuple[str, str, int]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['img1', 'img2', 'label'])
        for a, b, y in pairs:
            w.writerow([a, b, int(y)])

def summarize(pairs: List[Tuple[str, str, int]]) -> None:
    total = len(pairs)
    pos = sum(1 for _, _, y in pairs if y == 1)
    neg = total - pos
    print('Pairs summary:')
    print(f'- total_pairs: {total}')
    print(f'- positives: {pos}')
    print(f'- negatives: {neg}')

def main():
    ap = argparse.ArgumentParser(description='Build image pairs CSV from TAMPAR test images using id_{num} in filenames.')
    ap.add_argument('--root_dir', type=str, default='data/TAMPAR/test', help='Root directory containing category subfolders with images')
    ap.add_argument('--out_csv', type=str, default='data/TAMPAR/pairs_test.csv', help='Path to output CSV')
    ap.add_argument('--neg_per_pos', type=float, default=1.0, help='Number of negative pairs to sample per positive pair')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for negative sampling')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root_dir)
    files = find_images(root)
    print(f'Found {len(files)} image files under {root}')
    groups = index_by_id(files, root)
    total_ids = len(groups)
    multi_ids = sum(1 for v in groups.values() if len(v) >= 2)
    total_images_indexed = sum(len(v) for v in groups.values())
    print(f'Indexed {total_images_indexed} images with an id_*, across {total_ids} ids ({multi_ids} with >=2 images)')

    pos_pairs = build_positive_pairs(groups)
    n_pos = len(pos_pairs)
    print(f'Built {n_pos} positive pairs')

    target_neg = int(round(args.neg_per_pos * n_pos))
    neg_pairs = build_negative_pairs(groups, target_neg, rng)
    n_neg = len(neg_pairs)
    if n_neg < target_neg:
        print(f'Warning: requested {target_neg} negatives, sampled {n_neg} unique negatives (limited by data).')

    # Merge and shuffle for variety
    all_pairs = pos_pairs + neg_pairs
    rng.shuffle(all_pairs)

    out_csv = Path(args.out_csv)
    write_pairs_csv(all_pairs, out_csv)
    summarize(all_pairs)
    print(f'Saved pairs CSV to: {out_csv}')

if __name__ == '__main__':
    main()