# Roadmap

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