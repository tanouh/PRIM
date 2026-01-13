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


# 25/11/2025

## Task summary
We structured the project into a modular Python package, added parallel support for contrastive and triplet loss, and created CLI scripts for training and dataset standardization. The goal is to enable technology-dedicated branches (data pipeline, contrastive, triplet, model, deployment) and keep notebooks as lightweight orchestrators referencing reusable modules.

## Key deliverables (baseline scaffolding)
- Documentation: [ARCHITECTURE.md](ARCHITECTURE.md:1)
- Core package:
  - [prim_package/__init__.py](prim_package/__init__.py:1)
  - [prim_package/models/siamese.py](prim_package/models/siamese.py:1)
  - [prim_package/training/losses.py](prim_package/training/losses.py:1)
  - [prim_package/training/engine.py](prim_package/training/engine.py:1)
  - [prim_package/data_processing/datasets.py](prim_package/data_processing/datasets.py:1)
  - [prim_package/data_processing/transforms.py](prim_package/data_processing/transforms.py:1)
- CLI scripts:
  - [scripts/train.py](scripts/train.py:1)
  - [scripts/generate_pairs.py](scripts/generate_pairs.py:1)
  - [scripts/generate_triplets.py](scripts/generate_triplets.py:1)

## Branching model (one branch per technology)
- main: stable baseline
- develop: integration branch
- feature/data-pipeline: data loaders, transforms, CSV generation
- feature/contrastive-loss: contrastive objective and training loop
- feature/triplet-loss: triplet objective and training loop
- feature/model-architecture: encoder/backbone variants
- feature/api-deployment: serving, packaging, Docker

## Current capabilities
- Contrastive training: pair datasets + ContrastiveLoss; validation reports pos/neg mean distance.
- Triplet training: triplet datasets + TripletLoss; validation reports mean d(a,p) and d(a,n).
- CSV standardization utilities for pairs and triplets with flexible column names.

## Next steps (execution roadmap)
1) Data
   - Consolidate pair CSVs using: `python scripts/generate_pairs.py --csv Code/tampar_pairs_ssl.csv --out Code/pairs_standard.csv`
   - Consolidate triplet CSVs using: `python scripts/generate_triplets.py --csv Code/tampar_triplets_ssl.csv --out Code/triplets_standard.csv`
   - Decide consistent root prefix (`--root_dir`) for relative paths.
2) Training
   - Contrastive: `python scripts/train.py --objective contrastive --csv Code/pairs_standard.csv --root_dir .`
   - Triplet: `python scripts/train.py --objective triplet --csv Code/triplets_standard.csv --root_dir . --margin 0.3`
   - Add checkpointing and early stopping if needed.
3) Post-processing (requested)
   - Implement a data post-processing script executed after training to produce artifacts needed downstream (e.g., embeddings export, distance histograms, calibrated thresholds).
   - Suggested future file: [scripts/postprocess_embeddings.py](scripts/postprocess_embeddings.py:1) to compute and store embeddings for a gallery and evaluate retrieval.
4) Configuration
   - Add [config.yaml](config.yaml:1) to capture training hyperparams; `scripts/train.py` already supports `--config` overrides.
5) Testing
   - Add unit tests under tests/ for datasets, losses, and engines.
6) Notebooks
   - Move exploratory notebooks to notebooks/ and import from prim_package; keep cells short and single-responsibility.

## Milestones
- M1: Data standardization reproducible (pairs/triplets CSVs) — owner: data
- M2: Contrastive baseline trained and saved — owner: model
- M3: Triplet baseline trained and saved — owner: model
- M4: Post-processing pipeline delivers artifacts (embeddings, metrics) — owner: pipeline
- M5: Tests and config-based runs — owner: eng

## Quick reference commands
- Init commit:
  - `git add ARCHITECTURE.md prim_package scripts && git commit -m "Scaffold modular ML architecture"`
- Create branches:
  - `git branch feature/data-pipeline`
  - `git branch feature/contrastive-loss`
  - `git branch feature/triplet-loss`
  - `git branch feature/model-architecture`
  - `git branch feature/api-deployment`

## Assumptions and notes
- TorchVision ResNet50 with IMAGENET1K_V2 weights when `--pretrained 1`.
- L2-normalized embeddings from the projection head, compatible with cosine distance.
- Seed set to 42 in training for reproducibility.
- Dataloaders default to num_workers=0 to avoid env issues; raise as needed.