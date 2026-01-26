Adding OCR = moving from unimodal (Image-only) to Multimodal (image + text) architecture. 

---
## General understanding & advantages

`EmbeddingNet` (ResNet50) looks at **visual** features (shape, color, texture). 
However, when comparing parcel receipts, visual similarity might not be sufficient. 

### The approach :
1. **Visual stream** Visual embedding extraction by ResNet50 ($V$)
2. **Text stream** OCR engine extractsraw text from the image. A text encoder (like BERT or a simple LSTM) converts that text into a semantic text embedding ($T$)
3. **Fusion** Concatenate $V$ and $T$ and pass through them a final fc layer to create a consolidated embedding $Z$

### Advantages
- **Disambiguation**: Two visually identical shipping labels with different addresses will now have large distances
- **Semantic matching**: illustration ~an image of a "cat" and an image of the word "cat" can theoretically be pulled closer (if the text model is strong enough)

## Libraries needed 
1. **OCR extraction (preprocessing)**
- paddleocr or easyocr : deep-learning based, high accuracy, supports gpu
2. **Text encoding (in-model)**
- transformers (Hugging Face): to use a pre-trained model like DistilBERT to convert words into vectors. 

## Implementation strategy
**Must not run OCR inside training loop** (`forward` method) OCR inference is slow and non-differentiable (cannot backpropagate through the OCR engine to improve the OCR itself)
The workflow:
1. **Step A (offline)**: run a script once to extract text from all images and save it to CSVs.
2. **Step B (Dataset)** modify the `dataset` to load the text string alongside the image
3. **Step C (Model)** add a text encoder branch to `EmbeddingNet`

