Tracking log — Kaggle parcel dataset preparation for Siamese baseline
Date: 2025-11-17

Objective
- Prepare the Kaggle “Parcel_Detection” dataset for a first ResNet50 + Siamese similarity baseline.
- Produce consolidated labels and a pairs CSV suitable for self-supervised similarity training/evaluation.
- Keep outputs reproducible and colocated under data/.

Inputs and locations
- Kaggle dataset (Roboflow export, Pascal VOC):
  - Root: [data/kaggle](data/kaggle:1)
  - Readmes: [data/kaggle/README.dataset.txt](data/kaggle/README.dataset.txt:1), [data/kaggle/README.roboflow.txt](data/kaggle/README.roboflow.txt:1)
  - Splits: train/, valid/
  - Labels: VOC .xml per image inside each split
  - License: CC BY 4.0 (from README)
- TAMPAR: present under [data/TAMPAR](data/TAMPAR:1) (sample structure observed), will be integrated later for supervised “same parcel” pairs.

Work performed
1) Consolidated VOC annotations to a single CSV
- Script: [Code/scripts/prepare_kaggle_labels.py](Code/scripts/prepare_kaggle_labels.py:1)
- Output CSV: [data/kaggle/annotations.csv](data/kaggle/annotations.csv:1)
- Schema columns: split, image_filename, image_path, annotation_path, image_id, bbox_id, width, height, class, xmin, ymin, xmax, ymax, area
- Summary (from execution):
  - splits_found: train, valid
  - total_images: 202
  - total_boxes: 671
  - classes_per_split: single class “Boxes” in both splits

2) Generated Siamese-ready pairs CSV (self-supervised positives + negatives)
- Script: [Code/scripts/generate_kaggle_pairs_csv.py](Code/scripts/generate_kaggle_pairs_csv.py:1)
- Output CSV: [data/kaggle/pairs_kaggle_ssl.csv](data/kaggle/pairs_kaggle_ssl.csv:1)
- Pair generation logic:
  - Positives: “ssl_same_bbox” (same bbox twice; downstream data loader should apply different augmentations on-the-fly)
  - Negatives: “neg_different_image” (random bbox from a different image_id)
- Summary (from execution):
  - total_pairs: 1342
  - positives: 671
  - negatives: 671
  - by_split: train=1320, valid=22

Quality checks
- All Kaggle images are 640×640 (per VOC headers and README).
- Single detection class “Boxes” (no identity labels across images). Suitable for detection and self-supervised similarity; supervised identity will come from TAMPAR.
- No crops materialized; bbox fields retained in CSV for on-the-fly cropping (saves space and enables stronger augmentation).

How to reproduce locally
- Using the existing venv:
  1) Build annotations CSV
     "Code\envdev\Scripts\python.exe" "Code\scripts\prepare_kaggle_labels.py" --root "data\kaggle" --out_csv "data\kaggle\annotations.csv"
  2) Build pairs CSV
     "Code\envdev\Scripts\python.exe" "Code\scripts\generate_kaggle_pairs_csv.py" --annotations_csv "data\kaggle\annotations.csv" --out_pairs_csv "data\kaggle\pairs_kaggle_ssl.csv" --neg_per_pos 1

Notes and next steps
- Kaggle provides detection labels only; supervised “same parcel” identity pairs will be built from TAMPAR once parcel_id folder structure is provided.
- Next: wire the pairs CSV into the baseline notebook [Code/SiameseNetwork.ipynb](Code/SiameseNetwork.ipynb:1) to:
  - Load pairs_kaggle_ssl.csv
  - Crop by bbox at runtime, extract ResNet50 embeddings, compute cosine similarity
  - Report ROC-AUC/EER; calibrate threshold
- After TAMPAR integration: generate identity positives/negatives and create a mixed pairs CSV for fine-tuning the Siamese model.

Artifacts produced
- Annotations: [data/kaggle/annotations.csv](data/kaggle/annotations.csv:1)
- Pairs (SSL): [data/kaggle/pairs_kaggle_ssl.csv](data/kaggle/pairs_kaggle_ssl.csv:1)
- Scripts:
  - [Code/scripts/prepare_kaggle_labels.py](Code/scripts/prepare_kaggle_labels.py:1)
  - [Code/scripts/generate_kaggle_pairs_csv.py](Code/scripts/generate_kaggle_pairs_csv.py:1)
---

Tracking log — TAMPAR dataset preparation for Siamese baseline
Date: 2025-11-17

Objective
- Prepare the TAMPAR dataset to mirror the Kaggle preparation for the ResNet50 + Siamese similarity baseline.
- Produce consolidated labels and a pairs CSV suitable for self-supervised similarity training/evaluation.
- Keep outputs reproducible and colocated under data/.

Inputs and locations
- TAMPAR (COCO format):
  - Root: [data/TAMPAR](data/TAMPAR:1)
  - README: [data/TAMPAR/README.md](data/TAMPAR/README.md:1)
  - Annotations (COCO JSON): [data/TAMPAR/tampar_validation.json](data/TAMPAR/tampar_validation.json:1), [data/TAMPAR/tampar_test.json](data/TAMPAR/tampar_test.json:1)
  - Splits: validation, test (from COCO JSON filenames)
  - License: CC BY 4.0 (per README)
  - Note: image files may be inside tampar.zip; scripts do not require reading images, only metadata paths are built consistently.

Work performed
1) Consolidated COCO annotations to a single CSV
- Script: [Code/scripts/prepare_tampar_labels.py](Code/scripts/prepare_tampar_labels.py:1)
- Output CSV: [data/TAMPAR/annotations_tampar.csv](data/TAMPAR/annotations_tampar.csv:1)
- Schema columns (parity with Kaggle): split, image_filename, image_path, annotation_path, image_id, bbox_id, width, height, class, xmin, ymin, xmax, ymax, area
- Summary (from execution):
  - splits_found: validation, test
  - total_images: 732
  - total_boxes: 732
  - images_per_split: validation=247, test=485
  - boxes_per_split: validation=247, test=485
  - classes_per_split: validation={normal box: 247}, test={normal box: 485}
  - classes_detected: ['normal box']

2) Generated Siamese-ready pairs CSV (self-supervised positives + negatives)
- Script: [Code/scripts/generate_tampar_pairs_csv.py](Code/scripts/generate_tampar_pairs_csv.py:1)
- Output CSV: [data/TAMPAR/pairs_tampar_ssl.csv](data/TAMPAR/pairs_tampar_ssl.csv:1)
- Pair generation logic (identical to Kaggle):
  - Positives: “ssl_same_bbox” (same bbox twice; downstream data loader should apply different augmentations on-the-fly)
  - Negatives: “neg_different_image” (random bbox from a different image_id)
- Summary (from execution):
  - total_pairs: 1464
  - positives: 732
  - negatives: 732
  - by_split: validation=494, test=970

Quality checks
- COCO JSON detected with keys: info, categories, images, annotations; bbox converted from [x, y, w, h] to (xmin, ymin, xmax, ymax) with clipping to [0, W/H].
- image_path constructed as data/TAMPAR/{split}/{file_name} when file_name has no subfolder; otherwise preserves subfolder from file_name. This mirrors Kaggle CSV path semantics.
- Categories present in JSON (2 total), but current annotations map to class “normal box” in both splits.
- No crops materialized; bbox fields retained for on-the-fly cropping and augmentation.

How to reproduce locally
- Using the existing venv:
  1) Build annotations CSV
     "Code\envdev\Scripts\python.exe" "Code\scripts\prepare_tampar_labels.py" --json_list "data\TAMPAR\tampar_validation.json;data\TAMPAR\tampar_test.json" --images_root "data\TAMPAR" --out_csv "data\TAMPAR\annotations_tampar.csv"
  2) Build pairs CSV
     "Code\envdev\Scripts\python.exe" "Code\scripts\generate_tampar_pairs_csv.py" --annotations_csv "data\TAMPAR\annotations_tampar.csv" --out_pairs_csv "data\TAMPAR\pairs_tampar_ssl.csv" --neg_per_pos 1

Notes and next steps
- This mirrors the Kaggle flow described above; for supervised “same parcel” identity pairs we will need parcel identity metadata or folder structure mapping images to parcel_id.
- After integrating identity labels (if/when available), extend the pairs generator to add:
  - Positive identity pairs: same parcel_id, different images
  - Hard negatives: different parcel_id but similar appearance
- Then create a mixed pairs CSV and wire it into the baseline notebook [Code/SiameseNetwork.ipynb](Code/SiameseNetwork.ipynb:1).

Artifacts produced
- Annotations (COCO→CSV): [data/TAMPAR/annotations_tampar.csv](data/TAMPAR/annotations_tampar.csv:1)
- Pairs (SSL): [data/TAMPAR/pairs_tampar_ssl.csv](data/TAMPAR/pairs_tampar_ssl.csv:1)
- Scripts:
  - [Code/scripts/prepare_tampar_labels.py](Code/scripts/prepare_tampar_labels.py:1)
  - [Code/scripts/generate_tampar_pairs_csv.py](Code/scripts/generate_tampar_pairs_csv.py:1)