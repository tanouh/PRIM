import json
from pathlib import Path
import os
from typing import Any

def inspect_json(path: Path):
    print(f"--- Inspecting: {path}")
    if not path.exists():
        print("NOT FOUND")
        return
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"type: {type(data).__name__}")
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"keys({len(keys)}): {keys[:20]}")
        # Common COCO-style fields
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        if isinstance(images, list):
            print(f"images: {len(images)}")
            if images:
                print(f"image[0] keys: {list(images[0].keys())}")
        if isinstance(annotations, list):
            print(f"annotations: {len(annotations)}")
            if annotations:
                print(f"ann[0] keys: {list(annotations[0].keys())}")
        if isinstance(categories, list):
            print(f"categories: {len(categories)}")
            if categories:
                print(f"cat[0] keys: {list(categories[0].keys())}")
    elif isinstance(data, list):
        print(f"list length: {len(data)}")
        if data and isinstance(data[0], dict):
            print(f"item[0] keys: {list(data[0].keys())}")

def list_dir(p: Path, limit: int = 5):
    if not p.exists():
        print(f"{p} not found")
        return
    files = sorted([x.name for x in p.iterdir() if x.is_file()])[:limit]
    print(f"{p} sample files ({len(files)}): {files}")

def main():
    root = Path("data/TAMPAR")
    inspect_json(root / "tampar_validation.json")
    inspect_json(root / "tampar_test.json")
    for sub in ["validation", "test", "unlabeled", "uvmaps"]:
        list_dir(root / sub, limit=5)

if __name__ == "__main__":
    main()