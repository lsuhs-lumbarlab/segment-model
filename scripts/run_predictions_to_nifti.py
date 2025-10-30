"""
Convert predictions to 3D NIfTI mask for 3D Slicer.

This script reconstructs 2D slice predictions back into 3D NIfTI masks.
Supports both single patient and batch processing.

Usage:
    # Batch mode - convert all patients
    python scripts/run_predictions_to_nifti.py
    
    # Single patient
    python scripts/run_predictions_to_nifti.py --patient_id patient_test001
    
    # With original image for exact affine
    python scripts/run_predictions_to_nifti.py --patient_id patient_test001 --original_image data/inference/patient_test001.nii.gz
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.yolo.predictions_to_nifti import reconstruct_3d_mask
import nibabel as nib


def find_patient_ids(predictions_dir: Path) -> list:
    """
    Find all unique patient IDs from prediction files.
    
    Args:
        predictions_dir: Directory containing prediction JSON files
        
    Returns:
        List of patient IDs
    """
    prediction_files = list(predictions_dir.glob("*_slice*.json"))
    
    if not prediction_files:
        return []
    
    # Extract unique patient IDs
    patient_ids = set()
    for pred_file in prediction_files:
        # Remove _sliceXXXX.json suffix
        name = pred_file.stem
        if "_slice" in name:
            patient_id = name.split("_slice")[0]
            patient_ids.add(patient_id)
    
    return sorted(patient_ids)


def find_original_image(patient_id: str, inference_dir: Path) -> Path:
    """
    Try to find the original NIfTI image for a patient.
    
    Args:
        patient_id: Patient identifier
        inference_dir: Directory where original images might be
        
    Returns:
        Path to original image or None
    """
    # Try common patterns
    patterns = [
        inference_dir / f"{patient_id}.nii.gz",
        inference_dir / patient_id / "image.nii.gz",  # Old structure
    ]
    
    for path in patterns:
        if path.exists():
            return path
    
    return None


def process_single_patient(
    patient_id: str,
    predictions_dir: Path,
    output_dir: Path,
    original_image: Path = None,
    inference_dir: Path = None,
):
    """Process a single patient's predictions to NIfTI."""
    
    print(f"\n{'=' * 70}")
    print(f"Processing: {patient_id}")
    print(f"{'=' * 70}")
    
    # Try to find original image if not provided
    if original_image is None and inference_dir is not None:
        original_image = find_original_image(patient_id, inference_dir)
        if original_image:
            print(f"Found original image: {original_image}")
    
    # Reconstruct 3D mask
    try:
        mask_3d, mask_nifti = reconstruct_3d_mask(
            predictions_dir,
            patient_id,
            original_image,
        )
        
        # Save NIfTI
        output_file = output_dir / f"{patient_id}_pred_mask.nii.gz"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        nib.save(mask_nifti, str(output_file))
        
        print(f"✓ Saved: {output_file.name}")
        print(f"  Shape: {mask_3d.shape}")
        print(f"  Labels: {list(set(mask_3d.flatten().tolist()))}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert predictions to 3D NIfTI masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch mode - process all patients
  python scripts/run_predictions_to_nifti.py
  
  # Single patient
  python scripts/run_predictions_to_nifti.py --patient_id patient_test001
  
  # With original image for exact affine/header
  python scripts/run_predictions_to_nifti.py --patient_id patient_test001 --original_image data/inference/patient_test001.nii.gz
  
  # Batch with custom paths
  python scripts/run_predictions_to_nifti.py --predictions outputs/inference/predictions --output outputs/masks
        """
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="Patient ID to process (if not provided, processes all patients)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/inference/predictions"),
        help="Directory containing prediction JSON files (default: outputs/inference/predictions)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/masks"),
        help="Output directory for NIfTI masks (default: outputs/masks)",
    )
    parser.add_argument(
        "--original_image",
        type=Path,
        default=None,
        help="Path to original NIfTI image (optional, for exact affine/header)",
    )
    parser.add_argument(
        "--inference_dir",
        type=Path,
        default=Path("data/inference"),
        help="Directory containing original NIfTI files (default: data/inference)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Convert Predictions to 3D NIfTI Masks")
    print("=" * 70)
    
    # Verify predictions directory exists
    if not args.predictions.exists():
        print(f"✗ ERROR: Predictions directory not found: {args.predictions}")
        print("\nHave you run inference yet? Try:")
        print("  python scripts/run_inference_test_patient.py")
        sys.exit(1)
    
    # Determine mode: single patient or batch
    if args.patient_id:
        # Single patient mode
        print(f"Mode: Single patient")
        print(f"Patient: {args.patient_id}")
        print(f"Predictions: {args.predictions.absolute()}")
        print(f"Output: {args.output.absolute()}")
        print("=" * 70)
        
        success = process_single_patient(
            args.patient_id,
            args.predictions,
            args.output,
            args.original_image,
            args.inference_dir,
        )
        
        if not success:
            sys.exit(1)
    
    else:
        # Batch mode
        print(f"Mode: Batch processing")
        print(f"Predictions: {args.predictions.absolute()}")
        print(f"Output: {args.output.absolute()}")
        
        # Find all patient IDs
        patient_ids = find_patient_ids(args.predictions)
        
        if not patient_ids:
            print(f"\n✗ ERROR: No prediction files found in {args.predictions}")
            print("\nExpected files like: patient_test001_slice0000.json")
            sys.exit(1)
        
        print(f"\nFound {len(patient_ids)} patients to process:")
        for pid in patient_ids:
            print(f"  - {pid}")
        print("=" * 70)
        
        # Process each patient
        results = {"success": [], "failed": []}
        
        for patient_id in patient_ids:
            success = process_single_patient(
                patient_id,
                args.predictions,
                args.output,
                None,  # Let it auto-detect
                args.inference_dir,
            )
            
            if success:
                results["success"].append(patient_id)
            else:
                results["failed"].append(patient_id)
        
        # Summary
        print("\n" + "=" * 70)
        print("Batch Processing Complete")
        print("=" * 70)
        print(f"Successfully processed: {len(results['success'])}/{len(patient_ids)}")
        
        if results["success"]:
            print(f"\n✓ Success ({len(results['success'])}):")
            for pid in results["success"]:
                print(f"  - {pid}_pred_mask.nii.gz")
        
        if results["failed"]:
            print(f"\n✗ Failed ({len(results['failed'])}):")
            for pid in results["failed"]:
                print(f"  - {pid}")
        
        print(f"\nOutput directory: {args.output.absolute()}")
        print("\nYou can now load these masks in 3D Slicer alongside the original images!")
        print("=" * 70)
        
        if results["failed"]:
            sys.exit(1)