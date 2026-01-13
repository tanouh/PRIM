# Project Architecture Plan

This document outlines the modular architecture for the Siamese Network project, designed to support multiple training objectives and facilitate technology-specific development branches.

### 1. Proposed Architecture: A Modular Python Project

The project will be restructured from a notebook-centric workflow to a Python package to improve code reusability, testing, and separation of concerns.

#### Directory Structure:

```
PRIM/
├── ARCHITECTURE.md        # This file
├── data/                  # Raw and processed data
├── notebooks/             # For experimentation and visualization
│   └── SiameseContrastiveReduced.ipynb
├── prim_package/          # The core Python package for project logic
│   ├── __init__.py
│   ├── data_processing/   # Modules for data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── datasets.py    # PairImageDataset and TripletImageDataset classes
│   │   └── transforms.py  # Data augmentation and normalization pipelines
│   ├── models/            # Model architecture definitions
│   │   ├── __init__.py
│   │   └── siamese.py     # SiameseNet and EmbeddingNet classes
│   ├── training/          # Training, evaluation, and loss functions
│   │   ├── __init__.py
│   │   ├── engine.py      # train_contrastive and train_triplet functions
│   │   └── losses.py      # ContrastiveLoss and TripletLoss classes
│   └── utils/             # Utility functions
│       ├── __init__.py
│       └── file_utils.py
├── scripts/               # Standalone scripts for key tasks
│   ├── generate_pairs.py
│   ├── generate_triplets.py # New script for triplet data generation
│   └── train.py           # Script to run training, selectable loss function
├── tests/                 # Unit tests for the package
│   ├── test_datasets.py
│   └── test_models.py
├── config.yaml            # Configuration file for training parameters
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

### 2. Git Branching Strategy

A feature-branching workflow will be used to isolate development of different components.

*   **`main`**: Stable, production-ready code.
*   **`develop`**: The main integration branch for new features.
*   **`feature/data-pipeline`**: For all work related to `prim_package/data_processing/`.
*   **`feature/model-architecture`**: For changes to the models in `prim_package/models/`.
*   **`feature/contrastive-loss`**: For the original contrastive training logic.
*   **`feature/triplet-loss`**: A new branch for implementing the triplet loss functionality.
*   **`feature/api-deployment`**: A potential future branch for building a REST API.

### 3. Architecture Diagram

This diagram visualizes the workflow, including parallel paths for both Contrastive and Triplet training.

```mermaid
graph TD
    subgraph "Data Preparation"
        A[Raw Image Data] --> B{scripts/generate_pairs.py};
        B --> C[Image Pairs CSV];
        A --> D{scripts/generate_triplets.py};
        D --> E[Image Triplets CSV];
    end

    subgraph "Core Logic (prim_package)"
        F[config.yaml] --> G{scripts/train.py};
        
        subgraph "Contrastive Path"
            C --> C_DS[data_processing.datasets.PairImageDataset];
            C_DS --> C_DL[PyTorch DataLoader];
            C_LOSS[training.losses.ContrastiveLoss] --> C_TRAIN[training.engine.train_contrastive];
        end

        subgraph "Triplet Path"
            E --> T_DS[data_processing.datasets.TripletImageDataset];
            T_DS --> T_DL[PyTorch DataLoader];
            T_LOSS[training.losses.TripletLoss] --> T_TRAIN[training.engine.train_triplet];
        end

        M[models.siamese] --> C_DL & T_DL;
        C_TRAIN & T_TRAIN --> SM[Saved Model Artifact];
    end

    subgraph "Experimentation"
        N[notebooks/*.ipynb] --> O[Calls functions from prim_package];
        O --> P[Visualizations & Analysis];
    end