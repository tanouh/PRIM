#### Extract OCR text from all images in a CSV dataset
#### Supply chain parcel verification: extract text for sender/receiver photos
import os
import argparse
import pandas as pd
import easyocr
from tqdm import tqdm
import time

def main():
    parser = argparse.ArgumentParser(description="Extract OCR text from images for supply chain verification")
    parser.add_argument("--csv", required=True, help="CSV with image_path,label,split columns (e.g., gallery_query.csv)")
    parser.add_argument("--root_dir", default="data", help="Root directory for relative image paths")
    parser.add_argument("--out", default="csv/ocr_texts.csv", help="Output CSV: image_path, label, split, ocr_text")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU for OCR")
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    # Load input CSV
    print(f"Loading image list from {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"Found {len(df)} images")

    # Initialize OCR Reader (GPU-accelerated)
    print("Loading EasyOCR model (this may take a moment)...")
    reader = easyocr.Reader(['en'], gpu=args.gpu)
    
    data = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    
    print(f"\nExtracting OCR text from {len(df)} images...")
    start_time = time.time()
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row['image_path']
        label = row['label']
        split = row['split']
        
        # Construct full path
        full_path = image_path if os.path.isabs(image_path) else os.path.join(args.root_dir, image_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            print(f"\n⚠️  File not found: {full_path}")
            ocr_text = ""
            confidence = 0.0
        else:
            # Check extension
            if os.path.splitext(full_path)[1].lower() not in valid_exts:
                ocr_text = ""
                confidence = 0.0
            else:
                try:
                    # Extract OCR text with confidence scores
                    result = reader.readtext(full_path, detail=1)  # detail=1 returns (bbox, text, confidence)
                    
                    if result:
                        texts = [item[1] for item in result]
                        confidences = [item[2] for item in result]
                        ocr_text = " ".join(texts)
                        confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    else:
                        ocr_text = ""
                        confidence = 0.0
                        
                except Exception as e:
                    print(f"\n⚠️  Failed to read {image_path}: {e}")
                    ocr_text = ""
                    confidence = 0.0
        
        data.append({
            "image_path": image_path,
            "label": label,
            "split": split,
            "ocr_text": ocr_text,
            "ocr_confidence": confidence
        })

    elapsed = time.time() - start_time
    
    # Save to CSV
    df_output = pd.DataFrame(data)
    df_output.to_csv(args.out, index=False)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"OCR Extraction Complete")
    print(f"{'='*60}")
    print(f"Total images processed: {len(df_output)}")
    print(f"Images with text extracted: {(df_output['ocr_text'].str.len() > 0).sum()}")
    print(f"Average OCR confidence: {df_output['ocr_confidence'].mean():.4f}")
    print(f"Execution time: {elapsed:.2f}s ({elapsed/len(df_output):.2f}s per image)")
    print(f"Output saved to: {args.out}")
    print(f"{'='*60}\n")
    
    # Show sample results
    print("Sample OCR Results:")
    for idx in range(min(3, len(df_output))):
        row = df_output.iloc[idx]
        text_preview = row['ocr_text'][:80] + "..." if len(row['ocr_text']) > 80 else row['ocr_text']
        print(f"  [{row['label']}] {row['image_path']}")
        print(f"    → Text: {text_preview}")
        print(f"    → Confidence: {row['ocr_confidence']:.4f}\n")

if __name__ == "__main__":
    main()