"""
Convert 2D slice predictions back to 3D NIfTI mask.

This script:
1. Loads all prediction polygons for a patient
2. Reconstructs them back into 3D volume
3. Uses geometry metadata to restore original spacing/orientation
4. Saves as NIfTI mask compatible with 3D Slicer
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm


def load_prediction_json(json_path: Path) -> Tuple[List[Dict], dict]:
    """
    Load predictions and geometry from JSON file.

    Returns:
        (detections, geometry_metadata)
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    detections = data.get("detections", [])
    geometry = data.get("geometry", {})
    
    return detections, geometry


def polygon_to_mask(
    polygon_pixels: List[float],
    image_shape: Tuple[int, int],
    class_id: int,
) -> np.ndarray:
    """
    Convert polygon coordinates to binary mask.

    Args:
        polygon_pixels: Flattened list [x1, y1, x2, y2, ...]
        image_shape: (height, width)
        class_id: Class ID to fill (1-4)

    Returns:
        Binary mask with class_id values
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Reshape to Nx2 array of points
    points = np.array(polygon_pixels).reshape(-1, 2).astype(np.int32)
    
    # Fill polygon with class_id
    cv2.fillPoly(mask, [points], class_id)
    
    return mask


def reconstruct_3d_mask(
    predictions_dir: Path,
    patient_id: str,
    original_image_path: Path = None,
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Reconstruct 3D mask from 2D slice predictions.

    Args:
        predictions_dir: Directory containing prediction JSON files
        patient_id: Patient identifier to filter slices
        original_image_path: Optional path to original NIfTI (to copy header/affine)

    Returns:
        (mask_3d_array, reference_nifti_image)
    """
    # Find all prediction files for this patient
    prediction_files = sorted(predictions_dir.glob(f"{patient_id}_slice*.json"))
    
    if not prediction_files:
        raise ValueError(f"No prediction files found for patient {patient_id}")
    
    print(f"Found {len(prediction_files)} prediction files for {patient_id}")
    
    # Load first file to get dimensions
    first_detections, first_geometry = load_prediction_json(prediction_files[0])
    
    if not first_geometry:
        raise ValueError(f"No geometry metadata found in {prediction_files[0]}")
    
    # Get volume dimensions from geometry
    slice_shape = tuple(first_geometry["image_shape"])  # (H, W)
    num_slices = len(prediction_files)
    
    # Determine axis from geometry
    axis = first_geometry.get("axis", 0)
    
    # Create 3D volume in RAS orientation (matching export_slices.py)
    if axis == 0:  # Sagittal
        volume_shape = (num_slices, slice_shape[0], slice_shape[1])
    elif axis == 1:  # Coronal
        volume_shape = (slice_shape[0], num_slices, slice_shape[1])
    else:  # Axial
        volume_shape = (slice_shape[0], slice_shape[1], num_slices)
    
    mask_3d = np.zeros(volume_shape, dtype=np.uint8)
    
    print(f"Reconstructing 3D mask with shape: {volume_shape}")
    
    # Process each slice
    for pred_file in tqdm(prediction_files, desc="Reconstructing slices"):
        detections, geometry = load_prediction_json(pred_file)
        
        slice_idx = geometry["slice_index"]
        
        # Create 2D mask for this slice
        slice_mask = np.zeros(slice_shape, dtype=np.uint8)
        
        for det in detections:
            class_id = det["class_id"] + 1  # Convert 0-indexed to 1-indexed (1-4)
            polygon = det["polygon_pixels"]
            
            # Draw this detection on slice mask
            poly_mask = polygon_to_mask(polygon, slice_shape, class_id)
            
            # Combine with existing mask (higher class IDs overwrite lower ones)
            slice_mask = np.maximum(slice_mask, poly_mask)
        
        # Rotate back (inverse of the rotation in export_slices.py)
        # export_slices.py did: np.rot90(slice, k=1)
        # So we need: np.rot90(slice, k=-1) or k=3
        slice_mask = np.rot90(slice_mask, k=-1)
        
        # Insert into 3D volume
        if axis == 0:  # Sagittal
            mask_3d[slice_idx, :, :] = slice_mask
        elif axis == 1:  # Coronal
            mask_3d[:, slice_idx, :] = slice_mask
        else:  # Axial
            mask_3d[:, :, slice_idx] = slice_mask
    
    # Create NIfTI image
    if original_image_path and original_image_path.exists():
        # Load original to get exact affine
        print(f"Loading original image for affine: {original_image_path}")
        original_nib = nib.load(str(original_image_path))
        
        # Reorient original to RAS (matching our reconstruction)
        from src.common.io_nifti import load_and_reorient_nifti
        _, original_ras = load_and_reorient_nifti(original_image_path, "RAS")
        
        affine = original_ras.affine
        header = original_ras.header.copy()
    else:
        # Use affine from geometry metadata
        print("Using affine from geometry metadata")
        affine = np.array(first_geometry["affine"])
        header = None
    
    # Create NIfTI image
    mask_nifti = nib.Nifti1Image(mask_3d.astype(np.int16), affine, header)
    
    return mask_3d, mask_nifti


def main():
    """CLI entry point for converting predictions to NIfTI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert 2D slice predictions to 3D NIfTI mask"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Directory containing prediction JSON files",
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        required=True,
        help="Patient identifier (e.g., patient_test001)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output NIfTI file path (e.g., outputs/masks/patient_test001_pred_mask.nii.gz)",
    )
    parser.add_argument(
        "--original_image",
        type=Path,
        default=None,
        help="Original NIfTI image (optional, for exact affine/header)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Convert Predictions to 3D NIfTI Mask")
    print("=" * 70)
    print(f"Patient ID: {args.patient_id}")
    print(f"Predictions: {args.predictions}")
    print(f"Output: {args.output}")
    
    # Verify inputs
    if not args.predictions.exists():
        print(f"✗ ERROR: Predictions directory not found: {args.predictions}")
        sys.exit(1)
    
    if args.original_image and not args.original_image.exists():
        print(f"✗ ERROR: Original image not found: {args.original_image}")
        sys.exit(1)
    
    try:
        # Reconstruct 3D mask
        mask_3d, mask_nifti = reconstruct_3d_mask(
            args.predictions,
            args.patient_id,
            args.original_image,
        )
        
        # Save NIfTI
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nib.save(mask_nifti, str(args.output))
        
        print("\n" + "=" * 70)
        print("Reconstruction Complete")
        print("=" * 70)
        print(f"3D mask shape: {mask_3d.shape}")
        print(f"Unique labels: {np.unique(mask_3d)}")
        print(f"  0 = background")
        print(f"  1 = vertebra")
        print(f"  2 = ivf")
        print(f"  3 = sacrum")
        print(f"  4 = spinal_canal")
        print(f"\nSaved to: {args.output.absolute()}")
        print("\nYou can now load this mask in 3D Slicer alongside the original image!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()