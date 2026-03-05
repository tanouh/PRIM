from __future__ import annotations

from typing import Tuple

import torch
from torchvision import transforms


def get_train_transforms(im_size: int = 256) -> transforms.Compose:
    """
    Data augmentation + normalization for training.
    """
    return transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # slight color jitter
        transforms.RandomRotation(10),  # rotate between -10 to +10 degrees
        transforms.RandomResizedCrop(im_size, scale=(0.9, 1.0)), # random crop between 90% to 100% of the image
        transforms.RandomGrayscale(p=0.2), # 20% chance to turn the image grayscale
        transforms.RandomInvert(p=0.1), # 10% chance to invert colors
        transforms.PILToTensor(),                    # uint8 [0,255]
        transforms.ConvertImageDtype(torch.float32), # float32 [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(im_size: int = 256) -> transforms.Compose:
    """
    Deterministic preprocessing + normalization for evaluation.
    """
    return transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])