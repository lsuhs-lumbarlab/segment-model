"""
NIfTI I/O utilities with spatial harmonization.

This module loads NIfTI medical images and label maps, ensures they are in RAS
orientation, and verifies spatial alignment (shape, spacing, origin).
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

import nibabel as nib
import numpy as np


def load_and_reorient_nifti(
    nifti_path: Path, target_orientation: str = "RAS"
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load a NIfTI file and reorient to target orientation (default: RAS).

    Args:
        nifti_path: Path to .nii or .nii.gz file
        target_orientation: Target orientation code (default: "RAS")

    Returns:
        data: Reoriented numpy array
        img_reoriented: Reoriented nibabel image object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be loaded
    """
    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    try:
        img = nib.load(str(nifti_path))
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI file {nifti_path}: {e}")

    # Get current orientation
    current_ornt = nib.aff2axcodes(img.affine)
    print(f"  Current orientation: {''.join(current_ornt)}")

    # Reorient to target
    img_reoriented = nib.as_closest_canonical(img)
    
    # If not already in target orientation, apply transform
    target_ornt = nib.orientations.axcodes2ornt(target_orientation)
    current_ornt_codes = nib.orientations.axcodes2ornt(current_ornt)
    
    if not np.array_equal(current_ornt_codes, target_ornt):
        # Calculate transform from current to target
        transform = nib.orientations.ornt_transform(current_ornt_codes, target_ornt)
        img_reoriented = img.as_reoriented(transform)
        print(f"  Reoriented to: {target_orientation}")
    else:
        print(f"  Already in {target_orientation} orientation")

    data = img_reoriented.get_fdata()
    return data, img_reoriented


def verify_spatial_alignment(
    img1: nib.Nifti1Image,
    img2: nib.Nifti1Image,
    name1: str = "image",
    name2: str = "label",
    tolerance: float = 1e-3,
) -> bool:
    """
    Verify that two NIfTI images are spatially aligned.

    Checks:
    - Shape match
    - Spacing (pixdim) match within tolerance
    - Affine matrix match within tolerance

    Args:
        img1: First nibabel image
        img2: Second nibabel image
        name1: Name for first image (for error messages)
        name2: Name for second image (for error messages)
        tolerance: Tolerance for floating-point comparisons

    Returns:
        True if aligned, False otherwise
    """
    aligned = True

    # Check shape
    if img1.shape != img2.shape:
        print(f"  ✗ Shape mismatch: {name1} {img1.shape} != {name2} {img2.shape}")
        aligned = False
    else:
        print(f"  ✓ Shape match: {img1.shape}")

    # Check spacing (pixdim)
    spacing1 = img1.header.get_zooms()[:3]
    spacing2 = img2.header.get_zooms()[:3]
    spacing_diff = np.abs(np.array(spacing1) - np.array(spacing2))

    if np.any(spacing_diff > tolerance):
        print(
            f"  ✗ Spacing mismatch: {name1} {spacing1} != {name2} {spacing2} "
            f"(diff: {spacing_diff})"
        )
        aligned = False
    else:
        print(f"  ✓ Spacing match: {spacing1} mm")

    # Check affine
    affine_diff = np.abs(img1.affine - img2.affine)
    if np.any(affine_diff > tolerance):
        print(f"  ✗ Affine mismatch (max diff: {affine_diff.max():.6f})")
        print(f"    {name1} affine:\n{img1.affine}")
        print(f"    {name2} affine:\n{img2.affine}")
        aligned = False
    else:
        print(f"  ✓ Affine match (max diff: {affine_diff.max():.6f})")

    return aligned


def print_nifti_info(img: nib.Nifti1Image, name: str = "Image"):
    """Print summary information about a NIfTI image."""
    print(f"\n{name} Info:")
    print(f"  Shape: {img.shape}")
    print(f"  Data type: {img.get_data_dtype()}")
    print(f"  Spacing: {img.header.get_zooms()[:3]} mm")
    print(f"  Orientation: {''.join(nib.aff2axcodes(img.affine))}")
    
    data = img.get_fdata()
    print(f"  Value range: [{data.min():.2f}, {data.max():.2f}]")
    
    if name.lower() == "label" or "label" in name.lower():
        unique_vals = np.unique(data)
        print(f"  Unique label values: {unique_vals}")


def main():
    """CLI entry point for checking NIfTI image/label pair alignment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and verify NIfTI image and label spatial alignment"
    )
    parser.add_argument(
        "--check",
        nargs=2,
        metavar=("IMAGE", "LABEL"),
        required=True,
        help="Paths to image.nii.gz and label.nii.gz",
    )
    parser.add_argument(
        "--orientation",
        default="RAS",
        help="Target orientation (default: RAS)",
    )

    args = parser.parse_args()
    image_path = Path(args.check[0])
    label_path = Path(args.check[1])

    print("=" * 60)
    print("NIfTI Spatial Alignment Check")
    print("=" * 60)

    # Load and reorient image
    print(f"\n[1/3] Loading image: {image_path.name}")
    try:
        img_data, img_reoriented = load_and_reorient_nifti(
            image_path, args.orientation
        )
        print_nifti_info(img_reoriented, "Image")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Load and reorient label
    print(f"\n[2/3] Loading label: {label_path.name}")
    try:
        label_data, label_reoriented = load_and_reorient_nifti(
            label_path, args.orientation
        )
        print_nifti_info(label_reoriented, "Label")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Verify alignment
    print(f"\n[3/3] Verifying spatial alignment...")
    aligned = verify_spatial_alignment(
        img_reoriented, label_reoriented, "image", "label"
    )

    print("\n" + "=" * 60)
    if aligned:
        print("✓ PASS: Image and label are spatially aligned")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ FAIL: Image and label are NOT spatially aligned")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()