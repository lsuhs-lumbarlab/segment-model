"""
Export sagittal slices from image-only NIfTI files (no masks) for inference.

Similar to export_slices.py but works with images that don't have masks.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Import from our common module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.common.io_nifti import load_and_reorient_nifti
from src.yolo.export_slices import normalize_intensity


def export_test_slices(
    patient_id: str,
    image_path: Path,
    output_dir: Path,
    axis: int = 0,  # Sagittal in RAS orientation
) -> int:
    """
    Export sagittal slices from a single patient's NIfTI image (no mask).

    Args:
        patient_id: Patient identifier (e.g., "patient_test001")
        image_path: Path to image.nii.gz
        output_dir: Root output directory
        axis: Slice axis (0=sagittal in RAS, 1=coronal, 2=axial)

    Returns:
        Number of slices exported
    """
    # Create output directories
    images_dir = output_dir / "images"
    meta_dir = output_dir / "meta"

    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Load and reorient to RAS
    print(f"\n  Loading {patient_id}...")
    img_data, img_nib = load_and_reorient_nifti(image_path, "RAS")

    # Normalize image intensity
    img_normalized = normalize_intensity(img_data)

    # Get geometry info
    spacing = img_nib.header.get_zooms()[:3]
    origin = img_nib.affine[:3, 3]
    affine = img_nib.affine.tolist()

    # Export slices along specified axis
    num_slices = img_normalized.shape[axis]
    slice_count = 0

    print(f"  Exporting {num_slices} sagittal slices...")
    for slice_idx in range(num_slices):
        # Extract slice
        if axis == 0:  # Sagittal
            img_slice = img_normalized[slice_idx, :, :]
        elif axis == 1:  # Coronal
            img_slice = img_normalized[:, slice_idx, :]
        else:  # Axial
            img_slice = img_normalized[:, :, slice_idx]

        # Rotate for proper orientation
        img_slice = np.rot90(img_slice, k=1)

        # Create slice identifier
        slice_name = f"{patient_id}_slice{slice_idx:04d}"

        # Save image as PNG
        img_out_path = images_dir / f"{slice_name}.png"
        cv2.imwrite(str(img_out_path), img_slice)

        # Save geometry metadata as JSON
        geometry = {
            "patient_id": patient_id,
            "slice_index": int(slice_idx),
            "axis": axis,
            "axis_name": ["sagittal", "coronal", "axial"][axis],
            "spacing_mm": [float(s) for s in spacing],
            "origin_ras": [float(o) for o in origin],
            "affine": affine,
            "image_shape": list(img_slice.shape),
        }

        meta_out_path = meta_dir / f"{slice_name}.json"
        with open(meta_out_path, "w") as f:
            json.dump(geometry, f, indent=2)

        slice_count += 1

    print(f"  ✓ Exported {slice_count} slices for {patient_id}")
    return slice_count


def process_batch_inference_files(
    input_dir: Path,
    output_dir: Path,
    axis: int = 0,
) -> dict:
    """
    Process all NIfTI files in the inference directory.

    Args:
        input_dir: Directory containing .nii.gz files
        output_dir: Output directory for slices
        axis: Slice axis (0=sagittal in RAS, 1=coronal, 2=axial)

    Returns:
        Dictionary with processing statistics
    """
    # Find all .nii.gz files
    nifti_files = sorted(input_dir.glob("*.nii.gz"))
    
    if not nifti_files:
        print(f"ERROR: No .nii.gz files found in {input_dir}")
        return {"processed": 0, "total_slices": 0}
    
    print(f"Found {len(nifti_files)} NIfTI files to process:")
    for nf in nifti_files:
        print(f"  - {nf.name}")
    
    stats = {"processed": 0, "total_slices": 0, "files": {}}
    
    for nifti_file in nifti_files:
        # Derive patient ID from filename (remove .nii.gz extension)
        patient_id = nifti_file.stem.replace(".nii", "")
        
        print(f"\n{'=' * 70}")
        print(f"Processing: {nifti_file.name} (ID: {patient_id})")
        print(f"{'=' * 70}")
        
        try:
            n_slices = export_test_slices(
                patient_id,
                nifti_file,
                output_dir,
                axis,
            )
            stats["processed"] += 1
            stats["total_slices"] += n_slices
            stats["files"][patient_id] = {"slices": n_slices, "status": "success"}
            
        except Exception as e:
            print(f"  ✗ ERROR processing {nifti_file.name}: {e}")
            stats["files"][patient_id] = {"slices": 0, "status": "failed", "error": str(e)}
            continue
    
    return stats


def main():
    """CLI entry point for exporting test slices."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export sagittal slices from image-only NIfTI for inference"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input NIfTI image file OR directory containing .nii.gz files (default: data/inference)",
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="Patient identifier (only needed if --input is a single file)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/inference/slices"),
        help="Output directory for slices (default: data/inference/slices)",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Slice axis: 0=sagittal, 1=coronal, 2=axial (default: 0)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Export Test Slices for Inference")
    print("=" * 70)

    # If no input specified, use default inference directory
    if args.input is None:
        args.input = Path("data/inference")
    
    # Check if input is a file or directory
    if not args.input.exists():
        print(f"ERROR: Input not found: {args.input}")
        sys.exit(1)
    
    try:
        if args.input.is_file():
            # Single file mode (backwards compatibility)
            if args.patient_id is None:
                # Derive patient ID from filename
                args.patient_id = args.input.stem.replace(".nii", "")
                print(f"Auto-detected patient ID: {args.patient_id}")
            
            n_slices = export_test_slices(
                args.patient_id,
                args.input,
                args.output_dir,
                args.axis,
            )

            print("\n" + "=" * 70)
            print("Export Complete")
            print("=" * 70)
            print(f"Slices exported: {n_slices}")
            print(f"Output directory: {args.output_dir.absolute()}")
            print("=" * 70)
        
        else:
            # Batch directory mode
            print(f"Batch mode: Processing all .nii.gz files in {args.input}")
            print("=" * 70)
            
            stats = process_batch_inference_files(
                args.input,
                args.output_dir,
                args.axis,
            )
            
            print("\n" + "=" * 70)
            print("Batch Export Complete")
            print("=" * 70)
            print(f"Files processed: {stats['processed']}")
            print(f"Total slices exported: {stats['total_slices']}")
            print(f"\nPer-file results:")
            for patient_id, info in stats["files"].items():
                status_symbol = "✓" if info["status"] == "success" else "✗"
                print(f"  {status_symbol} {patient_id}: {info['slices']} slices ({info['status']})")
            print(f"\nOutput directory: {args.output_dir.absolute()}")
            print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()