"""
Core package for the project.

Public API re-exports main building blocks for convenience:
- Datasets (pairs/triplets), split/load utilities
- Models (Siamese/Embedding)
- Training engine (contrastive/triplet) and losses
"""

from .data_processing.datasets import (
    PairImageDataset,
    TripletImageDataset,
    SingleImageDataset,
    load_single_df,
    load_pair_dfs,
    load_triplet_dfs,
    get_split,
)
from .models.siamese import EmbeddingNet, SiameseNet
from .training.engine import (
    train_contrastive,
    validate_contrastive,
    train_triplet,
    validate_triplet,
)
from .training.losses import ContrastiveLoss, TripletLoss, pairwise_distance

__all__ = [
    # data
    "PairImageDataset",
    "SingleImageDataset",
    "TripletImageDataset",
    "load_pair_dfs",
    "load_triplet_dfs",
    "load_single_df",
    "get_split",
    # models
    "EmbeddingNet",
    "SiameseNet",
    # training
    "train_contrastive",
    "validate_contrastive",
    "train_triplet",
    "validate_triplet",
    # losses
    "ContrastiveLoss",
    "TripletLoss",
    "pairwise_distance",
]