#### Normalize drive/ dataset filenames
#### Add parcel ID prefix to all images for consistent naming
#### Input: data/drive/id_100/IMG_20240904_184145_453.jpg
#### Output: data/drive/id_100/id_100_IMG_20240904_184145_453.jpg
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def normalize_filenames(drive_dir="data/drive", dry_run=False):
    """
    Rename all images in drive/ to have parcel ID prefix.
    
    Args:
        drive_dir: Path to drive directory
        dry_run: If True, show what would be done without doing it
    """
    drive_path = Path(drive_dir)
    
    if not drive_path.exists():
        print(f"❌ Directory not found: {drive_dir}")
        return
    
    print(f"{'='*70}")
    print(f"Normalizing filenames in {drive_dir}/")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"{'='*70}\n")
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    
    total_renamed = 0
    total_skipped = 0
    
    # Process each parcel folder
    parcel_folders = sorted([d for d in drive_path.iterdir() if d.is_dir()])
    
    for parcel_dir in tqdm(parcel_folders, desc="Processing parcel folders"):
        parcel_id = parcel_dir.name
        
        # Get all image files
        image_files = [f for f in parcel_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in valid_exts]
        
        for img_file in image_files:
            old_name = img_file.name
            
            # Check if already has prefix
            if old_name.startswith(f"{parcel_id}_"):
                total_skipped += 1
                continue
            
            # Create new name with parcel ID prefix
            new_name = f"{parcel_id}_{old_name}"
            new_path = parcel_dir / new_name
            
            if dry_run:
                print(f"  [{parcel_id}] {old_name} → {new_name}")
            else:
                try:
                    img_file.rename(new_path)
                    total_renamed += 1
                except Exception as e:
                    print(f"❌ Error renaming {old_name}: {e}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Normalization {'Simulation' if dry_run else 'Complete'}")
    print(f"{'='*70}")
    print(f"Total renamed: {total_renamed}")
    print(f"Already normalized: {total_skipped}")
    print(f"Total processed: {total_renamed + total_skipped}")
    
    if dry_run:
        print(f"\n✓ Dry run complete. Run with --execute to apply changes.")
    else:
        print(f"\n✓ All filenames normalized!")
    
    print(f"{'='*70}\n")
    
    # Show examples
    if total_renamed > 0 or total_skipped > 0:
        print("Examples of normalized filenames:")
        example_count = 0
        for parcel_dir in drive_path.iterdir():
            if parcel_dir.is_dir():
                for img_file in parcel_dir.glob("*"):
                    if img_file.is_file() and img_file.suffix.lower() in valid_exts:
                        if img_file.name.startswith(parcel_dir.name + "_"):
                            print(f"  ✓ {parcel_dir.name}/{img_file.name}")
                            example_count += 1
                            if example_count >= 5:
                                break
                if example_count >= 5:
                    break


def main():
    parser = argparse.ArgumentParser(
        description="Normalize drive/ dataset filenames with parcel ID prefix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without applying
  python scripts/normalize_filenames.py --dry-run
  
  # Apply normalization
  python scripts/normalize_filenames.py --execute
  
  # Custom drive directory
  python scripts/normalize_filenames.py --drive path/to/drive --execute
        """
    )
    parser.add_argument("--drive", default="data/drive", help="Path to drive directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--execute", action="store_true", help="Apply normalization")
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        print("⚠️  No action specified!")
        print("Run with --dry-run to preview, or --execute to apply normalization")
        return
    
    normalize_filenames(args.drive, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
