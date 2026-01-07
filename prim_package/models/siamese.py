from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from transformers import AutoModel, AutoConfig

class EmbeddingNet(nn.Module):
    """
    ResNet50 backbone (feature extractor) + MLP projection head.
    Output embeddings are L2-normalized.
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        # TorchVision ResNet50; use weights arg for >=0.13 API
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        # Remove classifier head
        self.backbone.fc = nn.Identity()
        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)


class SiameseNet(nn.Module):
    """
    Siamese wrapper around the shared EmbeddingNet encoder.
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.encoder = EmbeddingNet(embed_dim=embed_dim, pretrained=pretrained)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2
    

# Multimodal embedding models
class MultimodalEmbeddingNet(nn.Module):
    """
    Two-stream architecture 
    1. Image branch: ResNet50
    2. Text branch: DistilBERT
    3. Fusion: Concatenation -> MLP
    """
    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()

        # --- Image branch ---
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        img_in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Project images features to 512
        self.img_proj = nn.Linear(img_in_features, 512)

        # --- Text branch ---
        # Freezing text encoder 
        self.text_model_name = "distilbert-base-uncased"
        self.text_encoder = AutoModel.from_pretrained(self.text_model_name)
        text_in_features = self.text_encoder.config.hidden_size  # usually 768

        # Project text features to 512
        self.text_proj = nn.Linear(text_in_features, 512)

        # --- Fusion MLP ---
        # Input is 512 (img) + 512 (text) = 1024
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
        )

    def forward(self, img: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, C, H, W)
            input_ids: (B, Seq_Len) from tokenizer
            attention_mask: (B, Seq_Len) from tokenizer
        """
        # 1. Image Features
        x_img = self.backbone(img)      # (B, 2048)
        x_img = self.img_proj(x_img)    # (B, 512)
        
        # 2. Text Features
        # DistilBERT output[0] is (B, Seq_Len, 768). 
        # We take the first token [CLS] (index 0) for sentence embedding.
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        x_txt = txt_out.last_hidden_state[:, 0, :] # (B, 768)
        x_txt = self.text_proj(x_txt)              # (B, 512)

        # 3. Fusion
        combined = torch.cat((x_img, x_txt), dim=1) # (B, 1024)
        x_final = self.fusion(combined)
        
        return F.normalize(x_final, p=2, dim=1)


class MultimodalSiameseNet(nn.Module):
    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.encoder = MultimodalEmbeddingNet(embed_dim=embed_dim, pretrained=pretrained)

    def forward_once(self, img, input_ids, attention_mask):
        return self.encoder(img, input_ids, attention_mask)

    def forward(self, img1, text1, img2, text2):
        # text1/text2 are dicts containing 'input_ids' and 'attention_mask'
        z1 = self.forward_once(img1, text1['input_ids'], text1['attention_mask'])
        z2 = self.forward_once(img2, text2['input_ids'], text2['attention_mask'])
        return z1, z2