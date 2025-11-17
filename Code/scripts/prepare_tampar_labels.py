import argparse
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import os


FIELDNAMES_ANN = [
    "split",
    "image_filename",
    "image_path",
    "annotation_path",
    "image_id",
    "bbox_id",
    "width",
    "height",
    "class",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "area",
]


def infer_split_from_filename(json_path: Path) -> str:
    name = json_path.name.lower()
    if "validation" in name or "val" in name:
        return "validation"
    if "test" in name:
        return "test"
    # Fallback
    return "unknown"


def load_coco_json(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_int_bbox_xyxy(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    # Clip to image bounds and cast to int (consistent with VOC-style ints used in kaggle CSV)
    xmin = max(0, int(round(x)))
    ymin = max(0, int(round(y)))
    xmax = min(W, int(round(x + w)))
    ymax = min(H, int(round(y + h)))
    # Safety to avoid negatives or inverted
    xmax = max(xmin, xmax)
    ymax = max(ymin, ymax)
    return xmin, ymin, xmax, ymax


def build_image_path(images_root: Path, split: str, file_name: str) -> str:
    # If file_name already encodes a relative subfolder, respect it; otherwise prepend split/
    # This path may not exist locally if images are packed in tampar.zip, but we keep a consistent reference.
    fn = file_name.replace("\\", "/")
    if "/" in fn:
        return str((images_root / fn).as_posix())
    else:
        return str((images_root / split / fn).as_posix())


def collect_annotations_from_coco(
    json_path: Path,
    images_root: Path,
    forced_split: str = None,
) -> Tuple[List[Dict], Dict]:
    data = load_coco_json(json_path)
    split = forced_split or infer_split_from_filename(json_path)

    # Index helpers
    images_by_id = {img["id"]: img for img in data.get("images", [])}
    cats_by_id = {cat["id"]: cat for cat in data.get("categories", [])}

    rows: List[Dict] = []
    per_split_images = defaultdict(set)
    per_split_boxes = Counter()
    per_split_classes = defaultdict(Counter)

    bbox_counter_per_image = defaultdict(int)

    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        img = images_by_id.get(img_id)
        if img is None:
            # Skip if annotation references a non-existent image
            continue

        W = int(img["width"])
        H = int(img["height"])
        file_name = img["file_name"]
        image_filename = Path(file_name).name
        image_path = build_image_path(images_root, split, file_name)

        # COCO bbox is [x, y, w, h]
        bb = ann.get("bbox", [0, 0, 0, 0])
        x, y, w, h = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        xmin, ymin, xmax, ymax = to_int_bbox_xyxy(x, y, w, h, W, H)
        area = (xmax - xmin) * (ymax - ymin)

        cat = cats_by_id.get(ann.get("category_id"))
        cls_name = cat["name"] if cat and "name" in cat else str(ann.get("category_id"))

        bbox_id = bbox_counter_per_image[img_id]
        bbox_counter_per_image[img_id] += 1

        row = {
            "split": split,
            "image_filename": image_filename,
            "image_path": image_path,
            # keep a precise pointer to the source annotation inside the JSON
            "annotation_path": f"{json_path.as_posix()}#ann_id={ann.get('id')}",
            "image_id": str(img_id),
            "bbox_id": bbox_id,
            "width": W,
            "height": H,
            "class": cls_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "area": area,
        }
        rows.append(row)
        per_split_images[split].add(image_filename)
        per_split_boxes[split] += 1
        per_split_classes[split][cls_name] += 1

    summary = {
        "images_per_split": {k: len(v) for k, v in per_split_images.items()},
        "boxes_per_split": dict(per_split_boxes),
        "classes_per_split": {k: dict(v) for k, v in per_split_classes.items()},
        "total_images": sum(len(v) for v in per_split_images.values()),
        "total_boxes": sum(per_split_boxes.values()),
        "splits_found": [split],
        "source_json": json_path.as_posix(),
    }
    return rows, summary


def save_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES_ANN)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def merge_summaries(summaries: List[Dict]) -> Dict:
    images_per_split = defaultdict(int)
    boxes_per_split = Counter()
    classes_per_split = defaultdict(Counter)
    splits_found = []
    for s in summaries:
        for k, v in s.get("images_per_split", {}).items():
            images_per_split[k] += v
        for k, v in s.get("boxes_per_split", {}).items():
            boxes_per_split[k] += v
        for k, d in s.get("classes_per_split", {}).items():
            for c, n in d.items():
                classes_per_split[k][c] += n
        splits_found.extend(s.get("splits_found", []))
    summary = {
        "images_per_split": dict(images_per_split),
        "boxes_per_split": dict(boxes_per_split),
        "classes_per_split": {k: dict(v) for k, v in classes_per_split.items()},
        "total_images": sum(images_per_split.values()),
        "total_boxes": sum(boxes_per_split.values()),
        "splits_found": splits_found,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json_list",
        type=str,
        default="data/TAMPAR/tampar_validation.json;data/TAMPAR/tampar_test.json",
        help="Semicolon-separated list of COCO json files to consolidate.",
    )
    ap.add_argument(
        "--images_root",
        type=str,
        default="data/TAMPAR",
        help="Root directory used to build image_path. If file_name has no folder, split/ is prepended.",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="data/TAMPAR/annotations_tampar.csv",
        help="Output CSV path.",
    )
    args = ap.parse_args()

    images_root = Path(args.images_root)

    all_rows: List[Dict] = []
    all_summaries: List[Dict] = []

    json_paths = [Path(p.strip()) for p in args.json_list.split(";") if p.strip()]
    if not json_paths:
        raise ValueError("No JSON paths provided via --json_list")

    for jp in json_paths:
        if not jp.exists():
            print(f"WARNING: JSON not found, skipping: {jp}")
            continue
        rows, summary = collect_annotations_from_coco(jp, images_root)
        all_rows.extend(rows)
        all_summaries.append(summary)

    save_csv(all_rows, Path(args.out_csv))

    merged = merge_summaries(all_summaries)
    print("Summary:")
    print(f"- splits_found: {merged['splits_found']}")
    print(f"- total_images: {merged['total_images']}")
    print(f"- total_boxes: {merged['total_boxes']}")
    print("- images_per_split:", merged["images_per_split"])
    print("- boxes_per_split:", merged["boxes_per_split"])
    print("- classes_per_split:", merged["classes_per_split"])
    if merged["total_boxes"] == 0:
        print("WARNING: No boxes found.")
    unique_classes = set()
    for v in merged["classes_per_split"].values():
        unique_classes.update(v.keys())
    print(f"- classes_detected: {sorted(unique_classes)}")


if __name__ == "__main__":
    main()