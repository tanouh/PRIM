#### Multimodal Model: Visual (ResNet50) + Text (DistilBERT) fusion
#### For supply chain parcel verification
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------
# Text Encoder: DistilBERT for semantic text embeddings
# ---------------------------------------------------------
class TextEncoder(nn.Module):
    """
    Text encoder using DistilBERT pre-trained model.
    Maps OCR text → semantic embedding (384D)
    """
    def __init__(self, model_name="distilbert-base-uncased", max_length=512):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        
        # Load pre-trained DistilBERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT weights (optional; set to False for fine-tuning)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.output_dim = 384  # DistilBERT hidden size
    
    def forward(self, texts):
        """
        Args:
            texts: List of strings or single string
        
        Returns:
            embeddings: Tensor of shape (batch_size, 384)
        """
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to same device as model
        device = next(self.bert.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get BERT output
        with torch.no_grad():
            output = self.bert(**encoded, output_hidden_states=False)
        
        # Use [CLS] token embedding (first token)
        cls_embedding = output.last_hidden_state[:, 0, :]  # (batch_size, 384)
        
        # Normalize for better cosine similarity
        cls_embedding = F.normalize(cls_embedding, p=2, dim=1)
        
        return cls_embedding


# ---------------------------------------------------------
# Visual Encoder: ResNet50 for visual embeddings
# ---------------------------------------------------------
class VisualEncoder(nn.Module):
    """
    Visual encoder using ResNet50 backbone.
    Maps image → visual embedding (256D)
    """
    def __init__(self, embed_dim=256, pretrained=True):
        super().__init__()
        
        # ResNet50 backbone
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        
        # Remove classification head
        self.backbone.fc = nn.Identity()
        
        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
        )
        
        self.embed_dim = embed_dim
    
    def forward(self, x):
        """
        Args:
            x: Image tensor (batch_size, 3, H, W)
        
        Returns:
            embedding: Tensor (batch_size, embed_dim)
        """
        x = self.backbone(x)
        x = self.proj(x)
        # L2 normalize for cosine similarity
        return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------
# Fusion Layer: Combine visual and text embeddings
# ---------------------------------------------------------
class FusionLayer(nn.Module):
    """
    Fuse visual (256D) and text (384D) embeddings → final embedding (256D)
    
    Strategy: Concatenate + FC layers with residual path
    """
    def __init__(self, visual_dim=256, text_dim=384, output_dim=256):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        
        # Fusion MLP
        self.fusion_net = nn.Sequential(
            nn.Linear(visual_dim + text_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
        )
        
        # Projection of visual for residual connection
        self.visual_proj = nn.Linear(visual_dim, output_dim)
    
    def forward(self, visual_emb, text_emb):
        """
        Args:
            visual_emb: Tensor (batch_size, 256)
            text_emb: Tensor (batch_size, 384)
        
        Returns:
            fused_emb: Tensor (batch_size, 256)
        """
        # Concatenate
        concatenated = torch.cat([visual_emb, text_emb], dim=1)  # (batch_size, 640)
        
        # Fusion
        fused = self.fusion_net(concatenated)  # (batch_size, output_dim)
        
        # Residual connection from visual
        visual_proj = self.visual_proj(visual_emb)
        fused = fused + 0.3 * visual_proj  # Weighted residual
        
        # L2 normalize
        fused = F.normalize(fused, p=2, dim=1)
        
        return fused


# ---------------------------------------------------------
# Multimodal Embedding Network
# ---------------------------------------------------------
class MultimodalEmbeddingNet(nn.Module):
    """
    Combined visual + text embedding network for parcel verification.
    
    Inputs:
        - image: (batch_size, 3, H, W)
        - text: List of strings (OCR extracted text)
    
    Output:
        - embedding: (batch_size, embed_dim)
    """
    def __init__(self, visual_embed_dim=256, text_model="distilbert-base-uncased", 
                 pretrained=True, use_text=True):
        super().__init__()
        
        self.use_text = use_text
        
        # Visual encoder
        self.visual_encoder = VisualEncoder(embed_dim=visual_embed_dim, pretrained=pretrained)
        
        if use_text:
            # Text encoder
            self.text_encoder = TextEncoder(model_name=text_model, max_length=512)
            
            # Fusion layer
            self.fusion = FusionLayer(
                visual_dim=visual_embed_dim,
                text_dim=self.text_encoder.output_dim,
                output_dim=visual_embed_dim
            )
        
        self.embed_dim = visual_embed_dim
    
    def forward(self, images, texts=None):
        """
        Args:
            images: Tensor (batch_size, 3, H, W)
            texts: List of strings (batch_size,) - required if use_text=True
        
        Returns:
            embeddings: Tensor (batch_size, embed_dim)
        """
        # Visual embedding
        visual_emb = self.visual_encoder(images)  # (batch_size, visual_embed_dim)
        
        if self.use_text and texts is not None:
            # Text embedding
            text_emb = self.text_encoder(texts)  # (batch_size, 384)
            
            # Fuse
            fused_emb = self.fusion(visual_emb, text_emb)  # (batch_size, embed_dim)
            return fused_emb
        else:
            # Return visual only
            return visual_emb


# ---------------------------------------------------------
# Multimodal Siamese Network
# ---------------------------------------------------------
class MultimodalSiameseNet(nn.Module):
    """
    Siamese wrapper for multimodal embedding network.
    
    For supply chain: compares sender reference vs receiver photo
    
    Usage:
        model = MultimodalSiameseNet(use_text=True)
        
        # Sender reference encoding
        sender_emb = model.forward_once(sender_image, sender_text)
        
        # Receiver photo encoding
        receiver_emb = model.forward_once(receiver_image, receiver_text)
        
        # Full siamese forward (for training)
        z1, z2 = model(sender_image, receiver_image, sender_text, receiver_text)
    """
    def __init__(self, visual_embed_dim=256, text_model="distilbert-base-uncased",
                 pretrained=True, use_text=True):
        super().__init__()
        
        self.encoder = MultimodalEmbeddingNet(
            visual_embed_dim=visual_embed_dim,
            text_model=text_model,
            pretrained=pretrained,
            use_text=use_text
        )
        self.use_text = use_text
    
    def forward_once(self, images, texts=None):
        """
        Encode a single batch (sender reference or receiver photo)
        
        Args:
            images: Tensor (batch_size, 3, H, W)
            texts: List of strings (batch_size,) or None
        
        Returns:
            embeddings: Tensor (batch_size, embed_dim)
        """
        return self.encoder(images, texts)
    
    def forward(self, images1, images2, texts1=None, texts2=None):
        """
        Forward pass for training (processes both branches)
        
        Args:
            images1: Tensor (batch_size, 3, H, W) - sender/reference photos
            images2: Tensor (batch_size, 3, H, W) - receiver photos
            texts1: List of strings (batch_size,) - sender OCR texts (optional)
            texts2: List of strings (batch_size,) - receiver OCR texts (optional)
        
        Returns:
            z1: Embeddings from first branch (batch_size, embed_dim)
            z2: Embeddings from second branch (batch_size, embed_dim)
        """
        z1 = self.forward_once(images1, texts1)
        z2 = self.forward_once(images2, texts2)
        return z1, z2
