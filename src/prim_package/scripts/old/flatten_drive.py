#### Flatten drive/ dataset
#### Move all images from id_XXX/ subfolders into a single flat directory
#### Requires normalized filenames first (run normalize_filenames.py)
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def flatten_drive(src_dir="data/drive", out_dir="data/drive_flat", dry_run=False):
    """
    Flatten drive/ directory structure.
    Moves all images from id_XXX/ folders into a single directory.
    
    Args:
        src_dir: Source drive directory with id_XXX/ subfolders
        out_dir: Output flat directory
        dry_run: If True, preview without applying
    """
    src_path = Path(src_dir)
    out_path = Path(out_dir)
    
    if not src_path.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return
    
    print(f"{'='*70}")
    print(f"Flattening {src_dir}/ → {out_dir}/")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"{'='*70}\n")
    
    # Create output directory if not dry run
    if not dry_run:
        out_path.mkdir(parents=True, exist_ok=True)
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    
    total_copied = 0
    total_skipped = 0
    duplicates = 0
    
    # Collect all image files with their parcel ID
    parcel_folders = sorted([d for d in src_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(parcel_folders)} parcel folders\n")
    
    for parcel_dir in tqdm(parcel_folders, desc="Flattening directories"):
        parcel_id = parcel_dir.name
        
        # Get all image files
        image_files = sorted([f for f in parcel_dir.iterdir() 
                             if f.is_file() and f.suffix.lower() in valid_exts])
        
        for img_file in image_files:
            filename = img_file.name
            out_file = out_path / filename
            
            # Check for duplicates
            if out_file.exists():
                duplicates += 1
                if dry_run:
                    print(f"  ⚠️  DUPLICATE: {filename} (would skip)")
                continue
            
            if dry_run:
                # Just count in dry run
                total_copied += 1
            else:
                try:
                    shutil.copy2(img_file, out_file)
                    total_copied += 1
                except Exception as e:
                    print(f"❌ Error copying {filename}: {e}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Flattening {'Simulation' if dry_run else 'Complete'}")
    print(f"{'='*70}")
    print(f"Total images: {total_copied}")
    print(f"Duplicates found: {duplicates}")
    print(f"Total processed: {total_copied + total_skipped}")
    
    if dry_run:
        print(f"\n✓ Dry run complete. Run with --execute to apply.")
    else:
        print(f"\n✓ Directory flattened! All images in {out_dir}/")
    
    print(f"{'='*70}\n")
    
    # Show example files
    if total_copied > 0:
        print("Examples of flattened structure:")
        example_count = 0
        for f in sorted(out_path.glob("*")):
            if f.is_file() and example_count < 5:
                print(f"  ✓ {f.name}")
                example_count += 1


def main():
    parser = argparse.ArgumentParser(
        description="Flatten drive/ directory - move all images to single folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites:
  - Run normalize_filenames.py first to add parcel ID prefix
  
Examples:
  # Preview changes
  python scripts/flatten_drive.py --dry-run
  
  # Apply flattening
  python scripts/flatten_drive.py --execute
  
  # Custom paths
  python scripts/flatten_drive.py --src data/drive --out data/drive_flat --execute
        """
    )
    parser.add_argument("--src", default="data/drive", help="Source drive directory")
    parser.add_argument("--out", default="data/drive_flat", help="Output flat directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without applying")
    parser.add_argument("--execute", action="store_true", help="Apply flattening")
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        print("⚠️  No action specified!")
        print("Run with --dry-run to preview, or --execute to apply")
        return
    
    flatten_drive(args.src, args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
