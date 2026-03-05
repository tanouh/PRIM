#### Image Segmentation using YOLOv8
#### Segment parcels from background, save cleaned images
#### Use with ResNet50 visual encoder only
import os
import argparse
import pandas as pd
import cv2
import numpy as np
import torch
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
        
        # Get masks and boxes. Explicitly use tensor data to ensure numpy arrays downstream.
        masks = results[0].masks.data  # (N, H, W) torch tensor
        boxes = results[0].boxes.conf.cpu().numpy()
        
        # Use mask with highest confidence
        best_mask_idx = np.argmax(boxes)
        best_mask = masks[best_mask_idx].detach().cpu().numpy()
        best_mask = np.asarray(best_mask, dtype=np.float32)
        
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


def segment_with_maskrcnn(image_path, model, device, conf_threshold=0.5, mask_threshold=0.5):
    """
    Segment image using torchvision Mask R-CNN.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, False

        # Convert BGR -> RGB and to torch tensor in [0,1]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        with torch.no_grad():
            outputs = model([tensor.to(device)])

        if not outputs or len(outputs[0]["scores"]) == 0:
            return image, False

        scores = outputs[0]["scores"].detach().cpu().numpy()
        masks = outputs[0]["masks"].detach().cpu().numpy()  # (N, 1, H, W)

        keep = scores >= conf_threshold
        if not np.any(keep):
            return image, False

        # Use the highest-scoring mask
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx, 0]

        if best_mask.shape != image.shape[:2]:
            best_mask = cv2.resize(best_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        best_mask = (best_mask >= mask_threshold).astype(np.uint8)
        segmented = cv2.bitwise_and(image, image, mask=best_mask)

        return segmented, True

    except Exception as e:
        print(f"\n⚠️  Segmentation failed for {image_path}: {e}")
        return None, False


def main():
    parser = argparse.ArgumentParser(description="Segment parcel images using YOLOv8 or Mask R-CNN")
    parser.add_argument("--csv", required=True, help="CSV with image_path,label,split columns")
    parser.add_argument("--root_dir", default=".", help="Root directory for relative image paths")
    parser.add_argument("--out_dir", default="data/segmented", help="Output directory for segmented images")
    parser.add_argument("--out_csv", default="csv/gallery_query_segmented.csv", help="Output CSV mapping")
    parser.add_argument("--model", default="yolov8n-seg", help="YOLOv8 model size (yolov8n-seg, yolov8s-seg, etc)")
    parser.add_argument("--backend", choices=["yolov8", "maskrcnn"], default="yolov8", help="Segmentation backend")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Mask binarization threshold")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process (for testing)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) if os.path.dirname(args.out_csv) else ".", exist_ok=True)

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if args.gpu and device != "cuda":
        print("⚠️  CUDA not available; falling back to CPU.")

    # Load segmentation backend
    if args.backend == "yolov8":
        print("Loading YOLOv8 segmentation model (this may take a moment)...")
        try:
            from ultralytics import YOLO
            model = YOLO(f'{args.model}.pt')
            model.to(device)
            print(f"✓ Loaded {args.model} on {device}")
            segment_fn = lambda p: segment_with_yolov8(p, model, args.conf)
        except ImportError:
            print("⚠️  YOLOv8 not installed. Install with: pip install ultralytics")
            return
    else:
        print("Loading Mask R-CNN (torchvision)...")
        try:
            from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            model = maskrcnn_resnet50_fpn(weights=weights)
            model.eval()
            model.to(device)
            print(f"✓ Loaded Mask R-CNN on {device}")
            segment_fn = lambda p: segment_with_maskrcnn(p, model, device, args.conf, args.mask_threshold)
        except ImportError:
            print("⚠️  Torchvision not installed. Install with: pip install torchvision")
            return

    # Load input CSV
    print(f"Loading image list from {args.csv}...")
    df = pd.read_csv(args.csv)
    # df = df[df["label"] == "id_100"]
    
    # Limit for testing
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} images for testing")
    else:
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
        segmented_image, success = segment_fn(full_path)
        
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
