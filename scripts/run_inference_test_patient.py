"""
End-to-end inference on test patients.

This script:
1. Exports slices from image-only NIfTI files
2. Runs trained model inference
3. Saves predictions

Supports both:
- Single file: --input path/to/file.nii.gz
- Batch mode: --input path/to/directory/ (processes all .nii.gz files)
- Default batch: no --input (processes all files in data/inference/)
"""

import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode != 0:
        print(f"\n✗ ERROR: {description} failed")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference on test patients (single file or batch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch mode - process all .nii.gz files in data/inference/
  python scripts/run_inference_test_patient.py
  
  # Batch mode - process all .nii.gz files in custom directory
  python scripts/run_inference_test_patient.py --input path/to/nifti/files/
  
  # Single file mode
  python scripts/run_inference_test_patient.py --input patient001.nii.gz
  
  # Single file with custom patient ID
  python scripts/run_inference_test_patient.py --input john-smith.nii.gz --patient_id john_smith
        """
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to NIfTI file OR directory containing .nii.gz files (default: data/inference/)",
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="Patient identifier (optional, auto-derived from filename if not provided)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to trained model (default: auto-detect from latest training run)",
    )
    parser.add_argument(
        "--output_slices",
        type=Path,
        default=Path("data/inference/slices"),
        help="Output directory for exported slices (default: data/inference/slices)",
    )
    parser.add_argument(
        "--output_predictions",
        type=Path,
        default=Path("outputs/inference"),
        help="Output directory for predictions (default: outputs/inference)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("End-to-End Inference Pipeline")
    print("=" * 70)

    # Set default input directory if not specified
    if args.input is None:
        args.input = Path("data/inference")
        print(f"Using default input directory: {args.input}")
    
    # Determine mode
    if args.input.is_file():
        mode = "single"
        print(f"Mode: Single file")
        print(f"Input: {args.input}")
        if args.patient_id:
            print(f"Patient ID: {args.patient_id}")
        else:
            derived_id = args.input.stem.replace(".nii", "")
            print(f"Patient ID: {derived_id} (auto-derived)")
    elif args.input.is_dir():
        mode = "batch"
        nifti_count = len(list(args.input.glob("*.nii.gz")))
        print(f"Mode: Batch processing")
        print(f"Input directory: {args.input}")
        print(f"Found: {nifti_count} .nii.gz files")
    else:
        print(f"✗ ERROR: Input not found: {args.input}")
        sys.exit(1)
    
    print("=" * 70)

    # Auto-detect model if not specified
    if args.model is None:
        runs_dir = project_root / "outputs" / "runs"
        if runs_dir.exists():
            train_runs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("train_")])
            if train_runs:
                latest_run = train_runs[-1]
                args.model = latest_run / "weights" / "best.pt"
                print(f"Auto-detected model: {args.model}")
            else:
                print("✗ ERROR: No training runs found in outputs/runs/")
                sys.exit(1)
        else:
            print("✗ ERROR: outputs/runs/ directory not found")
            sys.exit(1)

    if not args.model.exists():
        print(f"✗ ERROR: Model not found: {args.model}")
        sys.exit(1)

    # Step 1: Export slices
    export_cmd = [
        "python", "-m", "src.yolo.export_test_slices",
        "--input", str(args.input),
        "--output_dir", str(args.output_slices),
    ]
    
    # Add patient_id only for single file mode if specified
    if mode == "single" and args.patient_id:
        export_cmd.extend(["--patient_id", args.patient_id])
    
    run_command(export_cmd, "Step 1: Export slices from NIfTI")

    # Step 2: Run inference
    infer_cmd = [
        "python", "-m", "src.yolo.infer",
        "--model", str(args.model),
        "--images", str(args.output_slices / "images"),
        "--output", str(args.output_predictions),
        "--meta", str(args.output_slices / "meta"),
        "--format", "both",
    ]
    run_command(infer_cmd, "Step 2: Run model inference")

    print("\n" + "=" * 70)
    print("Inference Pipeline Complete!")
    print("=" * 70)
    print(f"Slices: {args.output_slices / 'images'}/")
    print(f"Predictions (YOLO): {args.output_predictions / 'labels'}/")
    print(f"Predictions (JSON): {args.output_predictions / 'predictions'}/")
    print("=" * 70)