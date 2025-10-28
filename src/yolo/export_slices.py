"""
Export 2D sagittal slices from NIfTI volumes for YOLO segmentation.

This script:
1. Loads image/mask pairs and reorients to RAS
2. Extracts sagittal slices (axis=0 in RAS)
3. Normalizes intensities using robust percentiles (1-99) to uint8
4. Saves slices as PNG files
5. Saves geometry metadata (spacing, origin, affine) as JSON
6. Splits data into train/val/test by patient
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# Import from our common module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.common.io_nifti import load_and_reorient_nifti, verify_spatial_alignment


def normalize_intensity(
    img_data: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """
    Normalize image intensities using robust percentile scaling.

    Args:
        img_data: Input image array
        lower_percentile: Lower percentile for clipping (default: 1.0)
        upper_percentile: Upper percentile for clipping (default: 99.0)

    Returns:
        Normalized uint8 array [0, 255]
    """
    p_low = np.percentile(img_data, lower_percentile)
    p_high = np.percentile(img_data, upper_percentile)

    # Clip and scale to [0, 1]
    img_clipped = np.clip(img_data, p_low, p_high)
    img_normalized = (img_clipped - p_low) / (p_high - p_low + 1e-8)

    # Convert to uint8 [0, 255]
    img_uint8 = (img_normalized * 255).astype(np.uint8)

    return img_uint8


def export_sagittal_slices(
    patient_id: str,
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    axis: int = 0,  # Sagittal in RAS orientation
    split: str = "train",
) -> int:
    """
    Export sagittal slices from a single patient's NIfTI volumes.

    Args:
        patient_id: Patient identifier (e.g., "patient001")
        image_path: Path to image.nii.gz
        mask_path: Path to mask.nii.gz
        output_dir: Root output directory (e.g., data/derivatives/yolo_sagittal)
        axis: Slice axis (0=sagittal in RAS, 1=coronal, 2=axial)
        split: Dataset split ("train", "val", or "test")

    Returns:
        Number of slices exported
    """
    # Create output directories
    images_dir = output_dir / "images" / split
    masks_dir = output_dir / "masks" / split  # Binary masks per class
    meta_dir = output_dir / "meta" / split

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Load and reorient to RAS
    print(f"\n  Loading {patient_id}...")
    img_data, img_nib = load_and_reorient_nifti(image_path, "RAS")
    mask_data, mask_nib = load_and_reorient_nifti(mask_path, "RAS")

    # Verify alignment
    if not verify_spatial_alignment(img_nib, mask_nib, "image", "mask"):
        raise ValueError(f"Image and mask are not aligned for {patient_id}")

    # Normalize image intensity
    img_normalized = normalize_intensity(img_data)

    # Get geometry info
    spacing = img_nib.header.get_zooms()[:3]
    origin = img_nib.affine[:3, 3]
    affine = img_nib.affine.tolist()

    # Get unique mask labels (excluding background=0)
    unique_labels = np.unique(mask_data)
    unique_labels = unique_labels[unique_labels > 0]
    print(f"  Found labels: {unique_labels.tolist()}")

    # Export slices along specified axis
    num_slices = img_normalized.shape[axis]
    slice_count = 0

    print(f"  Exporting {num_slices} sagittal slices...")
    for slice_idx in range(num_slices):
        # Extract slice
        if axis == 0:  # Sagittal
            img_slice = img_normalized[slice_idx, :, :]
            mask_slice = mask_data[slice_idx, :, :]
        elif axis == 1:  # Coronal
            img_slice = img_normalized[:, slice_idx, :]
            mask_slice = mask_data[:, slice_idx, :]
        else:  # Axial
            img_slice = img_normalized[:, :, slice_idx]
            mask_slice = mask_data[:, :, slice_idx]

        # Skip empty slices (no labels)
        if not np.any(mask_slice > 0):
            continue

        # Transpose to (H, W) if needed and flip for proper orientation
        img_slice = np.rot90(img_slice, k=1)  # Rotate 90 degrees
        mask_slice = np.rot90(mask_slice, k=1)

        # Create slice identifier
        slice_name = f"{patient_id}_slice{slice_idx:04d}"

        # Save image as PNG
        img_out_path = images_dir / f"{slice_name}.png"
        cv2.imwrite(str(img_out_path), img_slice)

        # Save masks per class as separate PNGs (for polygon conversion later)
        for label_id in unique_labels:
            class_mask = (mask_slice == label_id).astype(np.uint8) * 255
            mask_out_path = masks_dir / f"{slice_name}_class{int(label_id)}.png"
            cv2.imwrite(str(mask_out_path), class_mask)

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
            "labels_present": [int(lbl) for lbl in unique_labels if np.any(mask_slice == lbl)],
        }

        meta_out_path = meta_dir / f"{slice_name}.json"
        with open(meta_out_path, "w") as f:
            json.dump(geometry, f, indent=2)

        slice_count += 1

    print(f"  ✓ Exported {slice_count} non-empty slices for {patient_id}")
    return slice_count


def split_patients(
    patient_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split patient IDs into train/val/test sets.

    Args:
        patient_ids: List of patient identifiers
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for test (default: 0.15)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with keys "train", "val", "test" and patient ID lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    np.random.seed(random_seed)
    shuffled = np.random.permutation(patient_ids)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": shuffled[:n_train].tolist(),
        "val": shuffled[n_train : n_train + n_val].tolist(),
        "test": shuffled[n_train + n_val :].tolist(),
    }

    return splits


def main():
    """CLI entry point for exporting slices."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export sagittal slices from NIfTI volumes for YOLO training"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw"),
        help="Input directory containing patient folders (default: data/raw)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/derivatives/yolo_sagittal"),
        help="Output directory for slices (default: data/derivatives/yolo_sagittal)",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Slice axis: 0=sagittal, 1=coronal, 2=axial (default: 0)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Export Sagittal Slices for YOLO Segmentation")
    print("=" * 70)

    # Find all patient folders
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    patient_folders = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    patient_ids = [p.name for p in patient_folders]

    if not patient_ids:
        print(f"ERROR: No patient folders found in {input_dir}")
        sys.exit(1)

    print(f"\nFound {len(patient_ids)} patients: {patient_ids}")

    # Split patients
    splits = split_patients(
        patient_ids,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(splits['train'])} patients - {splits['train']}")
    print(f"  Val:   {len(splits['val'])} patients - {splits['val']}")
    print(f"  Test:  {len(splits['test'])} patients - {splits['test']}")

    # Export slices for each split
    total_slices = {"train": 0, "val": 0, "test": 0}

    for split_name, patient_list in splits.items():
        print(f"\n{'=' * 70}")
        print(f"Processing {split_name.upper()} set ({len(patient_list)} patients)")
        print(f"{'=' * 70}")

        for patient_id in tqdm(patient_list, desc=f"{split_name} patients"):
            patient_folder = input_dir / patient_id
            image_path = patient_folder / "image.nii.gz"
            mask_path = patient_folder / "mask.nii.gz"

            if not image_path.exists():
                print(f"  WARNING: Image not found for {patient_id}, skipping")
                continue

            if not mask_path.exists():
                print(f"  WARNING: Mask not found for {patient_id}, skipping")
                continue

            try:
                n_slices = export_sagittal_slices(
                    patient_id,
                    image_path,
                    mask_path,
                    args.output_dir,
                    args.axis,
                    split_name,
                )
                total_slices[split_name] += n_slices

            except Exception as e:
                print(f"  ERROR processing {patient_id}: {e}")
                continue

    # Summary
    print("\n" + "=" * 70)
    print("Export Summary")
    print("=" * 70)
    print(f"Train slices: {total_slices['train']}")
    print(f"Val slices:   {total_slices['val']}")
    print(f"Test slices:  {total_slices['test']}")
    print(f"Total slices: {sum(total_slices.values())}")
    print(f"\nOutput directory: {args.output_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()