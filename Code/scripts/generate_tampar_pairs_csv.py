import csv
import argparse
import random
from pathlib import Path
from collections import defaultdict

# Build Siamese "pairs" CSV from TAMPAR consolidated annotations (COCO->CSV)
# Mirrors Code/scripts/generate_kaggle_pairs_csv.py for parity with No-Code/track.md

FIELDNAMES_ANN = [
    "split","image_filename","image_path","annotation_path","image_id","bbox_id",
    "width","height","class","xmin","ymin","xmax","ymax","area"
]

FIELDNAMES_PAIR = [
    "split",
    # A box
    "a_image_path","a_image_id","a_bbox_id","a_xmin","a_ymin","a_xmax","a_ymax","a_width","a_height","a_area","a_class",
    # B box
    "b_image_path","b_image_id","b_bbox_id","b_xmin","b_ymin","b_xmax","b_ymax","b_width","b_height","b_area","b_class",
    # label and meta
    "label_same","pair_type"
]


def load_annotations(ann_csv: Path):
    rows = []
    with ann_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # type conversions consistent with kaggle flow
            r["bbox_id"] = int(r["bbox_id"])
            r["width"] = int(r["width"])
            r["height"] = int(r["height"])
            r["xmin"] = int(r["xmin"])
            r["ymin"] = int(r["ymin"])
            r["xmax"] = int(r["xmax"])
            r["ymax"] = int(r["ymax"])
            r["area"] = int(r["area"])
            rows.append(r)
    return rows


def group_by_split(rows):
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    return by_split


def index_by_image(rows):
    by_image = defaultdict(list)
    for r in rows:
        key = (r["image_id"])
        by_image[key].append(r)
    return by_image


def build_pairs_for_split(rows, k_neg_per_pos=1, rng=None):
    if rng is None:
        rng = random.Random(123)
    pairs = []

    # positives (self-supervised): same box twice
    for r in rows:
        pair = {
            "split": r["split"],
            "a_image_path": r["image_path"], "a_image_id": r["image_id"], "a_bbox_id": r["bbox_id"],
            "a_xmin": r["xmin"], "a_ymin": r["ymin"], "a_xmax": r["xmax"], "a_ymax": r["ymax"],
            "a_width": r["width"], "a_height": r["height"], "a_area": r["area"], "a_class": r["class"],
            "b_image_path": r["image_path"], "b_image_id": r["image_id"], "b_bbox_id": r["bbox_id"],
            "b_xmin": r["xmin"], "b_ymin": r["ymin"], "b_xmax": r["xmax"], "b_ymax": r["ymax"],
            "b_width": r["width"], "b_height": r["height"], "b_area": r["area"], "b_class": r["class"],
            "label_same": 1,
            "pair_type": "ssl_same_bbox",
        }
        pairs.append(pair)

    # negatives: different image
    by_image = index_by_image(rows)
    image_ids = list(by_image.keys())

    for r in rows:
        for _ in range(k_neg_per_pos):
            for _trial in range(10):
                other_img = rng.choice(image_ids)
                if other_img != r["image_id"]:
                    break
            other_r = rng.choice(by_image[other_img])

            pair = {
                "split": r["split"],
                "a_image_path": r["image_path"], "a_image_id": r["image_id"], "a_bbox_id": r["bbox_id"],
                "a_xmin": r["xmin"], "a_ymin": r["ymin"], "a_xmax": r["xmax"], "a_ymax": r["ymax"],
                "a_width": r["width"], "a_height": r["height"], "a_area": r["area"], "a_class": r["class"],
                "b_image_path": other_r["image_path"], "b_image_id": other_r["image_id"], "b_bbox_id": other_r["bbox_id"],
                "b_xmin": other_r["xmin"], "b_ymin": other_r["ymin"], "b_xmax": other_r["xmax"], "b_ymax": other_r["ymax"],
                "b_width": other_r["width"], "b_height": other_r["height"], "b_area": other_r["area"], "b_class": other_r["class"],
                "label_same": 0,
                "pair_type": "neg_different_image",
            }
            pairs.append(pair)

    return pairs


def save_pairs(pairs, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES_PAIR)
        writer.writeheader()
        for p in pairs:
            writer.writerow(p)


def summarize(pairs):
    total = len(pairs)
    pos = sum(1 for p in pairs if int(p["label_same"]) == 1)
    neg = total - pos
    by_split = defaultdict(int)
    by_type = defaultdict(int)
    for p in pairs:
        by_split[p["split"]] += 1
        by_type[p["pair_type"]] += 1
    print("Pairs summary:")
    print(f"- total_pairs: {total}")
    print(f"- positives: {pos}")
    print(f"- negatives: {neg}")
    print(f"- by_split: {dict(by_split)}")
    print(f"- by_type: {dict(by_type)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations_csv", type=str, default="data/TAMPAR/annotations_tampar.csv")
    ap.add_argument("--out_pairs_csv", type=str, default="data/TAMPAR/pairs_tampar_ssl.csv")
    ap.add_argument("--neg_per_pos", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    ann_csv = Path(args.annotations_csv)
    if not ann_csv.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {ann_csv}")

    rows = load_annotations(ann_csv)
    by_split = group_by_split(rows)

    all_pairs = []
    for split, split_rows in by_split.items():
        pairs_split = build_pairs_for_split(split_rows, k_neg_per_pos=args.neg_per_pos, rng=rng)
        all_pairs.extend(pairs_split)

    save_pairs(all_pairs, Path(args.out_pairs_csv))
    summarize(all_pairs)
    print(f"Saved pairs CSV to: {args.out_pairs_csv}")


if __name__ == "__main__":
    main()