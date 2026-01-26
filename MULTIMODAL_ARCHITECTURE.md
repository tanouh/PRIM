# Multimodal Architecture

## File Structure

Your multimodal components are distributed across three locations, each with a clear responsibility:

### 1. **Core Model Architectures** (`prim_package/models/multimodal.py`)
Contains the building blocks:
- `TextEncoder`: DistilBERT-based text encoder (OCR → 384D embedding)
- `VisualEncoder`: ResNet50-based visual encoder (image → 256D embedding)
- `FusionLayer`: Concatenate + MLP (512D + 384D → 256D)
- `MultimodalEmbeddingNet`: Complete visual+text encoder
- `MultimodalSiameseNet`: Siamese wrapper for training/inference

### 2. **Training Loops** (`prim_package/training/engine.py`)
Has multimodal training functions:
- `train_contrastive_multimodal()`: Contrastive loss training
- `train_triplet_multimodal()`: Triplet loss training
- Takes datasets that return `(img, text_tokens, img2, text_tokens2, label)`

### 3. **Inference/Testing** (`scripts/test.py`, `scripts/predict.py`)
Use the trained multimodal model:
- Load checkpoint
- Extract embeddings from image+text pairs
- Compute similarity scores
- Save results to CSV

## Data Flow

### Training
```
Dataset (images + OCR text)
    ↓
tokenize_text() → {input_ids, attention_mask}
    ↓
train_contrastive_multimodal()
    ├─ MultimodalSiameseNet.forward(img1, txt1_dict, img2, txt2_dict)
    │   └─ MultimodalEmbeddingNet
    │       ├─ VisualEncoder(img) → 256D
    │       ├─ TextEncoder(txt_dict) → 384D
    │       └─ FusionLayer(concat) → 256D
    ├─ Compute loss
    └─ Backprop
    ↓
Checkpoint saved (state_dict)
```

### Inference (Testing/Prediction)
```
Query image + OCR text
    ↓
model.forward_once(img, txt_dict)
    └─ MultimodalEmbeddingNet → 256D embedding
    ↓
Gallery image + OCR text
    ↓
model.forward_once(img, txt_dict)
    └─ MultimodalEmbeddingNet → 256D embedding
    ↓
cosine_similarity(query_emb, gallery_emb) → score
    ↓
CSV output
```

## Key Points

1. **Text Tokenization**: Happens once when creating datasets or loading data (not in model forward pass)
   - Uses DistilBERT tokenizer
   - Returns: `{input_ids, attention_mask}` dictionaries
   
2. **Batch Training**: engine.py expects pre-tokenized text from dataset
   - Input: `(img1, txt1_dict, img2, txt2_dict, label)`
   - No string-level tokenization in forward pass

3. **Why 3 files?**
   - **multimodal.py**: Model definitions (reusable architecture)
   - **engine.py**: Training loops (loss computation, optimization)
   - **scripts/**: Data loading & processing (specific to your test/train CSV formats)

## Next Steps

1. ✅ Core architecture defined
2. ⏳ Create multimodal training dataset (merge CSVs with OCR text)
3. ⏳ Adapt `train.py` to use multimodal training functions
4. ⏳ Run training with contrastive or triplet loss
5. ⏳ Run inference with `predict.py` on test set
6. ⏳ Calibrate threshold on validation set
    ocr_text='John Doe, 123 Main St, Tracking ABC123'
 ...


### Next Steps

1. **Test OCR extraction**:
   ```bash
   python scripts/extract_ocr.py --csv csv/gallery_query.csv --out csv/ocr_texts.csv
   ```

2. **Create training dataset** (multimodal pairs with text)
   - Dataset needs: `path_a`, `path_b`, `label`, `text_a`, `text_b`
   - Text will be tokenized during batch loading

3. **Train multimodal model**:
   ```bash
   python scripts/train.py --multimodal --csv csv/multimodal_pairs.csv
   ```

4. **Deploy verification**:
   ```python
   model = ParcelVerificationModel(model_path='best_model.pt')
   decision = model.verify(sender_img, sender_text, receiver_img, receiver_text)
   ```

### Key Files Modified

- ✅ `prim_package/models/multimodal.py` - Adapted to match existing architecture
- ✅ `scripts/extract_ocr.py` - Updated for gallery_query.csv format
- ℹ️ `prim_package/models/siamese.py` - Already has MultimodalEmbeddingNet & MultimodalSiameseNet
- ℹ️ `prim_package/training/engine.py` - Already has multimodal training functions

