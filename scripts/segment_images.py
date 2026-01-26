#### Image Segmentation using YOLOv8
#### Segment parcels from background, save cleaned images
#### Use with ResNet50 visual encoder only
import os
import argparse
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

def segment_with_yolov8(image_path, model, conf_threshold=0.5):
    """
    Segment image using YOLOv8-seg.
    
    Args:
        image_path: Path to input image
        model: YOLOv8 segmentation model
        conf_threshold: Confidence threshold for detection
    
    Returns:
        segmented_image: Image with only detected object (rest is black)
        success: Boolean indicating if segmentation succeeded
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, False
        
        # Inference
        results = model(image, conf=conf_threshold, verbose=False)
        
        if len(results) == 0 or results[0].masks is None:
            # No objects detected, return original
            return image, False
        
        # Get masks and boxes
        masks = results[0].masks.cpu().numpy()
        boxes = results[0].boxes.conf.cpu().numpy()
        
        # Use mask with highest confidence
        best_mask_idx = np.argmax(boxes)
        best_mask = masks[best_mask_idx]
        
        # Resize mask to match image dimensions
        if best_mask.shape != image.shape[:2]:
            best_mask = cv2.resize(best_mask, (image.shape[1], image.shape[0]))
        
        # Apply mask: keep only segmented region
        best_mask = (best_mask > 0.5).astype(np.uint8)
        segmented = cv2.bitwise_and(image, image, mask=best_mask)
        
        return segmented, True
        
    except Exception as e:
        print(f"\n⚠️  Segmentation failed for {image_path}: {e}")
        return None, False


def main():
    parser = argparse.ArgumentParser(description="Segment parcel images using YOLOv8")
    parser.add_argument("--csv", required=True, help="CSV with image_path,label,split columns")
    parser.add_argument("--root_dir", default="data", help="Root directory for relative image paths")
    parser.add_argument("--out_dir", default="data/segmented", help="Output directory for segmented images")
    parser.add_argument("--out_csv", default="csv/gallery_query_segmented.csv", help="Output CSV mapping")
    parser.add_argument("--model", default="yolov8n-seg", help="YOLOv8 model size (yolov8n-seg, yolov8s-seg, etc)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) if os.path.dirname(args.out_csv) else ".", exist_ok=True)

    # Load YOLOv8
    print("Loading YOLOv8 segmentation model (this may take a moment)...")
    try:
        from ultralytics import YOLO
        model = YOLO(f'{args.model}.pt')
        if args.gpu:
            model.to('cuda')
        print(f"✓ Loaded {args.model}")
    except ImportError:
        print("⚠️  YOLOv8 not installed. Install with: pip install ultralytics")
        return

    # Load input CSV
    print(f"Loading image list from {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"Found {len(df)} images to segment")

    data = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    
    print(f"\nSegmenting {len(df)} images...")
    start_time = time.time()
    successful = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row['image_path']
        label = row['label']
        split = row['split']
        
        # Construct full path
        full_path = image_path if os.path.isabs(image_path) else os.path.join(args.root_dir, image_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            print(f"\n⚠️  File not found: {full_path}")
            continue
        
        # Check extension
        if os.path.splitext(full_path)[1].lower() not in valid_exts:
            continue
        
        # Segment image
        segmented_image, success = segment_with_yolov8(full_path, model, args.conf)
        
        if segmented_image is None:
            continue
        
        # Create output path preserving directory structure
        # e.g., TAMPAR/validation/id_100/image.jpg → segmented/TAMPAR/validation/id_100/image.jpg
        rel_path = os.path.relpath(full_path, args.root_dir)
        out_path = os.path.join(args.out_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Save segmented image
        cv2.imwrite(out_path, segmented_image)
        
        # Store mapping
        data.append({
            "image_path": image_path,
            "label": label,
            "split": split,
            "segmented_path": os.path.join(args.out_dir, rel_path),
            "segmentation_success": success
        })
        
        if success:
            successful += 1

    elapsed = time.time() - start_time
    
    # Save output CSV
    if data:
        df_output = pd.DataFrame(data)
        df_output.to_csv(args.out_csv, index=False)
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"Segmentation Complete")
        print(f"{'='*60}")
        print(f"Total images processed: {len(df_output)}")
        print(f"Successfully segmented: {successful}")
        print(f"Failed/skipped: {len(df_output) - successful}")
        print(f"Execution time: {elapsed:.2f}s ({elapsed/len(df_output):.2f}s per image)")
        print(f"Output CSV saved to: {args.out_csv}")
        print(f"Segmented images saved to: {args.out_dir}/")
        print(f"{'='*60}\n")
    else:
        print("No images were successfully segmented.")


if __name__ == "__main__":
    main()
