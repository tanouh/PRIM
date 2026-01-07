# extract_ocr.py
import os
import argparse
import pandas as pd
import easyocr
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Scan folder and OCR all images.")
    parser.add_argument("--image_dir", required=True, help="Folder containing images")
    parser.add_argument("--out", default="images_with_text.csv", help="Output path for master index")
    args = parser.parse_args()

    # Initialize Reader (GPU=True is crucial for speed)
    print("Loading OCR model...")
    reader = easyocr.Reader(['en'], gpu=True)
    
    data = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    
    print(f"Scanning {args.image_dir}...")
    
    # Walk through directory recursively
    for root, _, files in os.walk(args.image_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in valid_exts:
                full_path = os.path.join(root, fname)
                
                try:
                    # detail=0 returns a list of strings found
                    result = reader.readtext(full_path, detail=0) 
                    text_content = " ".join(result) # Join words with space
                except Exception as e:
                    print(f"Failed to read {fname}: {e}")
                    text_content = ""
                
                # We store the full path to ensure we can match it later
                data.append({
                    "filepath": full_path,
                    "text_content": text_content
                })
                
                if len(data) % 50 == 0:
                    print(f"Processed {len(data)} images...", end='\r')

    df = pd.DataFrame(data)
    df.to_csv(args.out, index=False)
    print(f"\nDone! Saved OCR data for {len(df)} images to {args.out}")

if __name__ == "__main__":
    main()