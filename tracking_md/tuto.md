# Tutorial: Generating Triplets CSV

## Overview

To create triplet pairs for training or testing your Siamese network, use the triplet generation scripts in the `scripts/` folder.

## For Drive Dataset

Generate test triplets from the `data/drive` folder where objects are grouped by their ID (e.g., `id_100`, `id_501`):

```bash
python scripts/generate_drive_triplets_test.py --data_folder data/drive
```

### Options

- `--data_folder` (required): Path to the data folder containing object images grouped by ID folders
- `--output`: Output CSV path (default: `csv/drive_triplets_test.csv`)
- `--split`: Split name to use in the CSV (default: `test`)
- `--seed`: Random seed for reproducibility (default: `1337`)

### Example with custom output

```bash
python scripts/generate_drive_triplets_test.py \
    --data_folder data/drive \
    --output csv/my_triplets.csv \
    --split validation \
    --seed 42
```

## For TAMPAR Dataset

Generate triplets from the TAMPAR test set for self-supervised learning:

```bash
python scripts/generate_tampar_triplets_ssl.py
```

This script uses hardcoded paths (`data/TAMPAR/test`) and generates triplets split into train/validation sets at 90/10 ratio.

## CSV Format

The generated CSV files contain triplet pairs with the following columns:
- `split`: Dataset split name (train/test/validation)
- `path_anchor`: Path to the anchor image
- `positive_path`: Path to a positive sample (same object as anchor)
- `negative_path`: Path to a negative sample (different object)

### Example rows

```csv
split,path_anchor,positive_path,negative_path
test,data/drive/id_100/img1.jpg,data/drive/id_100/img2.jpg,data/drive/id_502/img1.jpg
test,data/drive/id_101/img1.jpg,data/drive/id_101/img3.jpg,data/drive/id_100/img5.jpg
```

## How It Works

For each object ID folder:
1. All images within the same ID folder are considered as the **same object**
2. **Positive pairs**: Two different images from the same ID folder
3. **Negative samples**: Images from different ID folders
4. **Triplets**: (anchor, positive, negative) where anchor and positive share the same ID, negative has a different ID

The script ensures:
- Only IDs with at least 2 images are included (to form positive pairs)
- Each positive pair gets one randomly selected negative
- Deterministic output when using the same seed
