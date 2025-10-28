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


def main():
    """CLI entry point for exporting test slices."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export sagittal slices from image-only NIfTI for inference"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input NIfTI image file (e.g., data/inference/patient_test001/image.nii.gz)",
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        required=True,
        help="Patient identifier (e.g., patient_test001)",
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

    # Verify input exists
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    try:
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

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()