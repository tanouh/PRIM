import csv
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


def parse_voc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    filename = root.find('filename').text
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bnd = obj.find('bndbox')
        xmin = int(float(bnd.find('xmin').text))
        ymin = int(float(bnd.find('ymin').text))
        xmax = int(float(bnd.find('xmax').text))
        ymax = int(float(bnd.find('ymax').text))
        objects.append(
            {
                "class": name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )
    return filename, width, height, objects


def collect_annotations(root_dir: Path):
    rows = []
    per_split_images = defaultdict(set)
    per_split_boxes = Counter()
    per_split_classes = defaultdict(Counter)
    splits = []
    for split in ["train", "valid", "test", "validation", "val"]:
        if (root_dir / split).exists():
            splits.append(split)
    if not splits:
        # fallback: scan root
        splits = ["."]
    for split in splits:
        xml_dir = root_dir / split
        for xml_path in xml_dir.glob("*.xml"):
            try:
                filename, width, height, objects = parse_voc_xml(xml_path)
            except Exception as e:
                print(f"WARNING: failed parsing {xml_path}: {e}")
                continue
            image_rel = f"{split}/{filename}" if split != "." else filename
            image_id = Path(filename).stem
            per_split_images[split].add(filename)
            for i, obj in enumerate(objects):
                row = {
                    "split": split,
                    "image_filename": filename,
                    "image_path": str((root_dir / image_rel).as_posix()),
                    "annotation_path": str(xml_path.as_posix()),
                    "image_id": image_id,
                    "bbox_id": i,
                    "width": width,
                    "height": height,
                    "class": obj["class"],
                    "xmin": obj["xmin"],
                    "ymin": obj["ymin"],
                    "xmax": obj["xmax"],
                    "ymax": obj["ymax"],
                    "area": (obj["xmax"] - obj["xmin"]) * (obj["ymax"] - obj["ymin"]),
                }
                rows.append(row)
                per_split_boxes[split] += 1
                per_split_classes[split][obj["class"]] += 1
    summary = {
        "images_per_split": {k: len(v) for k, v in per_split_images.items()},
        "boxes_per_split": dict(per_split_boxes),
        "classes_per_split": {k: dict(v) for k, v in per_split_classes.items()},
        "total_images": sum(len(v) for v in per_split_images.values()),
        "total_boxes": sum(per_split_boxes.values()),
        "splits_found": splits,
    }
    return rows, summary


def save_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="data/kaggle",
        help="Root directory containing Pascal VOC xmls under split dirs.",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="data/kaggle/annotations.csv",
        help="Path to output CSV.",
    )
    args = ap.parse_args()
    root_dir = Path(args.root)
    rows, summary = collect_annotations(root_dir)
    save_csv(rows, Path(args.out_csv))
    # Print summary
    print("Summary:")
    print(f"- splits_found: {summary['splits_found']}")
    print(f"- total_images: {summary['total_images']}")
    print(f"- total_boxes: {summary['total_boxes']}")
    print("- images_per_split:", summary["images_per_split"])
    print("- boxes_per_split:", summary["boxes_per_split"])
    print("- classes_per_split:", summary["classes_per_split"])
    # Sanity checks
    if summary["total_boxes"] == 0:
        print("WARNING: No boxes found.")
    # Common label name issues
    unique_classes = set()
    for v in summary["classes_per_split"].values():
        unique_classes.update(v.keys())
    if unique_classes == {"Boxes"}:
        print(
            "Note: Single class 'Boxes' detected. Suitable for parcel detection; instance identity is not provided."
        )
    else:
        print(f"Classes detected: {sorted(unique_classes)}")


if __name__ == "__main__":
    main()