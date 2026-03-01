"""
Core package for the project.

Public API re-exports main building blocks for convenience:
- Datasets (pairs/triplets), split/load utilities
- Models (Siamese/Embedding)
- Training engine (contrastive/triplet) and losses
"""

# from .data_processing.datasets import (
#     PairImageDataset,
#     TripletImageDataset,
#     SingleImageDataset,
#     load_single_df,
#     load_pair_dfs,
#     load_triplet_dfs,
#     get_split,
# )
from .data_processing.datasets_multimodal import (
    PairImageDatasetMultimodal,
    TripletImageDatasetMultimodal,
    load_pair_dfs,
    load_triplet_dfs,
    get_split,

)
from .models.siamese import EmbeddingNet, SiameseNet
from .models.multimodal import MultimodalEmbeddingNet, MultimodalSiameseNet
from .training.engine import (
    train_contrastive_multimodal,
    validate_contrastive_multimodal,
    train_triplet_multimodal,
    validate_triplet_multimodal,
)
from .training.losses import ContrastiveLoss, TripletLoss, pairwise_distance

__all__ = [
    # data
    "PairImageDatasetMultimodal",
    "TripletImageDatasetMultimodal",
    "load_pair_dfs",
    "load_triplet_dfs",
    "get_split",
    # models
    "EmbeddingNet",
    "SiameseNet",
    "MultimodalEmbeddingNet",
    "MultimodalSiameseNet",
    # training
    "train_contrastive_multimodal",
    "validate_contrastive_multimodal",
    "train_triplet_multimodal",
    "validate_triplet_multimodal",
    # losses
    "ContrastiveLoss",
    "TripletLoss",
    "pairwise_distance",
]