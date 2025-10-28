"""
Convert predictions to 3D NIfTI mask for 3D Slicer.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.yolo.predictions_to_nifti import main as convert_main


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert predictions to NIfTI mask"
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        required=True,
        help="Patient ID (e.g., patient_test001)",
    )
    parser.add_argument(
        "--original_image",
        type=Path,
        default=None,
        help="Path to original image.nii.gz (optional)",
    )
    
    args = parser.parse_args()
    
    # Set up paths
    predictions_dir = project_root / "outputs" / "inference" / "predictions"
    output_file = project_root / "outputs" / "masks" / f"{args.patient_id}_pred_mask.nii.gz"
    
    # Build command
    sys.argv = [
        "predictions_to_nifti",
        "--predictions", str(predictions_dir),
        "--patient_id", args.patient_id,
        "--output", str(output_file),
    ]
    
    if args.original_image:
        sys.argv.extend(["--original_image", str(args.original_image)])
    
    print("=" * 70)
    print("Converting Predictions to NIfTI Mask")
    print("=" * 70)
    print(f"Patient: {args.patient_id}")
    print(f"Predictions: {predictions_dir}")
    print(f"Output: {output_file}")
    print("=" * 70)
    print()
    
    # Run conversion
    convert_main()