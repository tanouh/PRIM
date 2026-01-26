#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ID_REGEX = re.compile(r"(id_[^_/]+)")


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        "Build gallery/query CSV using id_<something> identity rule"
    )
    p.add_argument("--root_dir", default="./data/", help="Root data directory (e.g. data/)")
    p.add_argument("--out_csv", default="./csv/gallery_query.csv")
    p.add_argument("--seed", type=int, default=42)

    # Split config per ID
    p.add_argument("--min_gallery", type=int, default=1)
    p.add_argument("--min_query", type=int, default=1)
    p.add_argument("--gallery_ratio", type=float, default=0.5)

    return p.parse_args()


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def extract_id(path: Path) -> str | None:
    """
    Extract id_<something> from the path (filename or folders).
    Returns None if no ID is found.
    """
    match = ID_REGEX.search(str(path))
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)

    root = Path(args.root_dir)
    assert root.exists(), f"Root dir not found: {root}"

    id_to_images: Dict[str, List[Path]] = {}

    # 1) Scan all images
    for img_path in root.rglob("*"):
        if not is_image(img_path):
            continue

        label = extract_id(img_path)
        if label is None:
            print(f"[SKIP] No id_ found in path: {img_path}")
            continue

        id_to_images.setdefault(label, []).append(img_path)

    rows = []

    # 2) Split per identity
    for label, images in sorted(id_to_images.items()):
        if len(images) < args.min_gallery + args.min_query:
            print(f"[SKIP] {label}: not enough images")
            continue

        random.shuffle(images)

        n = len(images)
        n_gallery = max(args.min_gallery, int(n * args.gallery_ratio))
        n_gallery = min(n_gallery, n - args.min_query)

        gallery = images[:n_gallery]
        query = images[n_gallery:]

        for img in gallery:
            rows.append([
                str(img.relative_to(root)),
                label,
                "gallery",
            ])

        for img in query:
            rows.append([
                str(img.relative_to(root)),
                label,
                "query",
            ])

    # 3) Write CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "split"])
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print(f"CSV written to: {args.out_csv}")
    print(f"Identities found: {len(id_to_images)}")
    print(f"Total images written: {len(rows)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
