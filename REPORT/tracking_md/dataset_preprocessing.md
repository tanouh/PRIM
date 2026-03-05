# Dataset Preprocessing Pipeline

## Overview

This document describes the standardized preprocessing pipeline for the **drive/** dataset to create a clean, normalized ML dataset structure.

The pipeline consists of two main steps:
1. **Normalize filenames** - Add parcel ID prefix to all images
2. **Flatten directory structure** - Consolidate all images into a single folder

## Motivation

### Problem
- **drive/** had inconsistent naming: parcel ID only in folder name, not in filename
- Example: `data/drive/id_100/IMG_20240904_184145_453.jpg` - no ID in filename
- This made it difficult to:
  - Identify images outside of folder context
  - Process images in parallel
  - Handle missing or corrupted folder structures

### Solution
Normalize all images to have parcel ID prefix in filename, then flatten into single directory:
- **Before**: `data/drive/id_100/IMG_20240904_184145_453.jpg`
- **After**: `data/drive/id_100_IMG_20240904_184145_453.jpg` (flat folder)

This approach keeps parcel information in filenames (consistent with TAMPAR dataset) while simplifying directory structure.

---

## Step 1: Normalize Filenames

**Script**: `scripts/normalize_filenames.py`

### Purpose
Add parcel ID prefix to all images so every file is self-identifying.

### Usage

**Preview changes (recommended first step):**
```bash
python scripts/normalize_filenames.py --dry-run
```

**Apply normalization:**
```bash
python scripts/normalize_filenames.py --execute
```

### What It Does

1. Iterates through each parcel folder (`id_100/`, `id_101/`, etc.)
2. For each image file, prepends parcel ID to filename
3. Skips already-normalized files (idempotent - safe to run multiple times)
4. Validates file extensions (`.jpg`, `.png`, `.bmp`, `.webp`, `.tif`)

### Example Output

```
Before:
  data/drive/id_100/IMG_20240904_184145_453.jpg
  data/drive/id_100/signal-2024-09-03-23-43-53-904.jpg
  data/drive/id_101/IMG_20251020_141617_BURST5.jpg

After:
  data/drive/id_100/id_100_IMG_20240904_184145_453.jpg
  data/drive/id_100/id_100_signal-2024-09-03-23-43-53-904.jpg
  data/drive/id_101/id_101_IMG_20251020_141617_BURST5.jpg
```

### Key Features
- ✅ Dry-run mode (preview without changes)
- ✅ Idempotent (safe to run multiple times)
- ✅ Shows before/after examples
- ✅ Skips invalid/corrupt files
- ✅ Progress bar with tqdm

### Statistics

After running on drive/ dataset:
- Parcel folders: 40+ (id_100-id_122, id_501-id_508)
- Total images: 1000+
- Time: ~1-2 seconds

---

## Step 2: Flatten Directory

**Script**: `scripts/flatten_drive.py`

### Purpose
Move all normalized images from `id_XXX/` subfolders into a single flat directory.

### Usage

**Preview flattening:**
```bash
python scripts/flatten_drive.py --dry-run
```

**Apply flattening:**
```bash
python scripts/flatten_drive.py --execute
```

### What It Does

1. Creates output directory (`data/drive_flat/`)
2. Copies all images from `id_XXX/` folders to flat directory
3. Detects and skips duplicates (same filename)
4. Preserves parcel ID in filenames

### Example Output

```
Before:
  data/drive/
  ├── id_100/
  │   ├── id_100_IMG_20240904_184145_453.jpg
  │   └── id_100_signal-2024-09-03-23-43-53-904.jpg
  ├── id_101/
  │   ├── id_101_IMG_20251020_141617_BURST5.jpg
  │   └── id_101_IMG_20251020_141617_BURST14.jpg
  └── ... (40+ folders)

After:
  data/drive_flat/
  ├── id_100_IMG_20240904_184145_453.jpg
  ├── id_100_signal-2024-09-03-23-43-53-904.jpg
  ├── id_101_IMG_20251020_141617_BURST5.jpg
  ├── id_101_IMG_20251020_141617_BURST14.jpg
  ├── id_102_...jpg
  └── ... (1000+ files, all in one folder)
```

### Key Features
- ✅ Dry-run mode (preview)
- ✅ Duplicate detection
- ✅ Preserves parcel ID prefix
- ✅ Progress bar
- ✅ Summary statistics

### Optional: Replace Original

After flattening verification:
```bash
rm -rf data/drive  # Delete original (or keep as backup)
mv data/drive_flat data/drive
```

---

## Complete Workflow

### Full Preprocessing Pipeline

```bash
# Step 1: Normalize filenames (add parcel ID prefix)
python scripts/normalize_filenames.py --dry-run
python scripts/normalize_filenames.py --execute

# Step 2: Flatten directory structure
python scripts/flatten_drive.py --dry-run
python scripts/flatten_drive.py --execute

# Step 3: Replace original (optional)
rm -rf data/drive_backup  # Create backup if needed
mv data/drive data/drive_backup
mv data/drive_flat data/drive
```

### Result
✅ All images have parcel ID prefix in filename  
✅ Flat directory structure (easy to process)  
✅ Self-identifying filenames (no folder context needed)  

---

## Dataset Comparison

### After Preprocessing - Naming Consistency

**TAMPAR dataset:**
```
data/TAMPAR/validation/table/
├── id_00_20230523_160510.jpg
├── id_01_20230523_160443.jpg
├── id_05_20230523_160604.jpg
```
- ID in filename: ✅
- Flat structure: ❌ (nested by material type)

**drive dataset (after preprocessing):**
```
data/drive/
├── id_100_IMG_20240904_184145_453.jpg
├── id_100_signal-2024-09-03-23-43-53-904.jpg
├── id_101_IMG_20251020_141617_BURST5.jpg
```
- ID in filename: ✅
- Flat structure: ✅

---

## Next Steps

### 1. Create Training Dataset
After preprocessing, use the normalized flat structure to create training CSVs:
```bash
python scripts/consolidate_data.py --drive data/drive --out data/clean
```

### 2. Extract OCR (Multimodal)
If using multimodal approach (vision + text):
```bash
python scripts/extract_ocr.py --csv csv/gallery_query.csv --out csv/ocr_texts.csv
```

### 3. Segment Images (Optional)
For improved model performance with background removal:
```bash
python scripts/segment_images.py --csv csv/gallery_query.csv --execute
```

### 4. Train Model
Use cleaned, preprocessed data for training:
```bash
python scripts/train.py --config config.yaml --data data/clean
```

---

## Troubleshooting

### Issue: Script says "File not found"
- Ensure `data/drive/` exists
- Check path format (use relative paths from project root)

### Issue: Some images not processed
- Check file extensions are in valid list: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`
- Verify images are readable (not corrupted): `file data/drive/*/*.jpg`

### Issue: Want to undo flattening
```bash
# Reverse not directly supported, but you can:
1. Keep backup: mv data/drive data/drive_backup
2. Recreate from backup by extracting parcel ID from filename
```

---

## Design Rationale

### Why Normalize First?
- Ensures every image is self-identifying
- Matches TAMPAR naming convention
- Allows future folder restructuring without losing parcel info

### Why Flatten After?
- Simpler directory structure for data pipelines
- No need to manage nested folders during training
- Easier parallel processing (all images in one directory)
- Still preserves parcel ID in filename

### Why Both Steps?
- **Separation of concerns**: Naming (Step 1) is independent from structure (Step 2)
- **Reversibility**: Can flatten/unflatten, but normalized names persist
- **Flexibility**: Can use normalized images in nested OR flat structure
- **Safety**: Dry-run mode lets you verify before committing

---

## Statistics

### drive/ Dataset (Actual)
- Parcel IDs: 40+ (id_100-id_122, id_501-id_508)
- Total images: 1000+
- Image formats: JPG, PNG
- Preprocessing time: ~2-5 minutes

### Files Created
- `scripts/normalize_filenames.py` - 140 lines
- `scripts/flatten_drive.py` - 160 lines

---

## References

Related scripts:
- `scripts/consolidate_data.py` - Create train/val/test splits
- `scripts/extract_ocr.py` - Extract text from images
- `scripts/segment_images.py` - Segment parcels from background
- `scripts/normalize_filenames.py` - Add parcel ID prefix (THIS DOCUMENT)
- `scripts/flatten_drive.py` - Flatten directory structure (THIS DOCUMENT)


🔄 Assigning splits (parcel-level)...

======================================================================
✅ Triplet generation complete
======================================================================
Saved 470 triplets to csv/triplets_data.csv

Split distribution:
  train       :    330 triplets
  validation  :     70 triplets
  test        :     70 triplets

Parcels per split:
  train       :  33 parcels
  validation  :   7 parcels
  test        :   7 parcels
======================================================================

======================================================================
✅ Pair generation complete
======================================================================
Saved 535 pairs to csv/pairs_data.csv

Split distribution:
  train       :    385 pairs
  validation  :     75 pairs
  test        :     75 pairs

Label distribution:
  positive (1):    235 pairs
  negative (0):    300 pairs

Parcels per split:
  train       :  42 parcels
  validation  :   9 parcels
  test        :   9 parcels