#### Flatten TAMPAR dataset
#### Extract files with 'id' in filename from nested TAMPAR_raw structure
#### Consolidate into flat TAMPAR/ directory
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def flatten_tampar(src_dir="data/TAMPAR_raw", out_dir="data/TAMPAR", dry_run=False):
    """
    Flatten TAMPAR dataset structure.
    Only copies files with 'id' in filename (e.g., id_00_*.jpg).
    Consolidates from nested material/split folders into single flat directory.
    
    Args:
        src_dir: Source TAMPAR_raw directory with nested structure
        out_dir: Output flat TAMPAR directory
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
    print(f"Filter: Only files with 'id' in filename")
    print(f"{'='*70}\n")
    
    # Create output directory if not dry run
    if not dry_run:
        out_path.mkdir(parents=True, exist_ok=True)
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    
    total_copied = 0
    total_skipped = 0
    total_no_id = 0
    duplicates = 0
    
    # Recursively find all image files
    all_files = list(src_path.rglob("*"))
    image_files = [f for f in all_files 
                   if f.is_file() and f.suffix.lower() in valid_exts]
    
    print(f"Found {len(image_files)} image files in nested structure\n")
    
    for img_file in tqdm(image_files, desc="Flattening TAMPAR"):
        filename = img_file.name
        
        # Filter: only files with 'id' in filename
        if "id" not in filename.lower():
            total_no_id += 1
            continue
        
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
                total_skipped += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Flattening {'Simulation' if dry_run else 'Complete'}")
    print(f"{'='*70}")
    print(f"Files with 'id' in name (copied): {total_copied}")
    print(f"Files without 'id' (skipped): {total_no_id}")
    print(f"Duplicates found: {duplicates}")
    print(f"Errors: {total_skipped}")
    print(f"Total input files: {len(image_files)}")
    
    if dry_run:
        print(f"\n✓ Dry run complete. Run with --execute to apply.")
    else:
        print(f"\n✓ TAMPAR flattened! All files with 'id' in {out_dir}/")
    
    print(f"{'='*70}\n")
    
    # Show example files
    if total_copied > 0:
        print("Examples of flattened files:")
        example_count = 0
        for f in sorted(out_path.glob("id_*")):
            if f.is_file() and example_count < 10:
                parcel_id = f.name.split("_")[0:2]  # Extract id_XX
                parcel_str = "_".join(parcel_id)
                print(f"  ✓ [{parcel_str}] {f.name}")
                example_count += 1


def main():
    parser = argparse.ArgumentParser(
        description="Flatten TAMPAR dataset - consolidate nested material/split folders to flat directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites:
  - Rename original TAMPAR to TAMPAR_raw: mv data/TAMPAR data/TAMPAR_raw
  
Examples:
  # Preview changes
  python scripts/flatten_tampar.py --dry-run
  
  # Apply flattening
  python scripts/flatten_tampar.py --execute
  
  # Custom paths
  python scripts/flatten_tampar.py --src data/TAMPAR_raw --out data/TAMPAR --execute

Workflow:
  1. mv data/TAMPAR data/TAMPAR_raw
  2. python scripts/flatten_tampar.py --dry-run
  3. python scripts/flatten_tampar.py --execute
        """
    )
    parser.add_argument("--src", default="data/TAMPAR_raw", help="Source TAMPAR_raw directory")
    parser.add_argument("--out", default="data/TAMPAR", help="Output flat TAMPAR directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without applying")
    parser.add_argument("--execute", action="store_true", help="Apply flattening")
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        print("⚠️  No action specified!")
        print("Run with --dry-run to preview, or --execute to apply")
        return
    
    flatten_tampar(args.src, args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
