"""
End-to-end inference on a single test patient.

This script:
1. Exports slices from image-only NIfTI
2. Runs trained model inference
3. Saves predictions
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
        description="Run inference on a single test patient"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to test patient image.nii.gz",
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        required=True,
        help="Patient identifier (e.g., patient_test001)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to trained model (default: auto-detect from latest training run)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("End-to-End Inference on Test Patient")
    print("=" * 70)
    print(f"Patient ID: {args.patient_id}")
    print(f"Input: {args.input}")
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
        "--patient_id", args.patient_id,
        "--output_dir", "data/inference/slices",
    ]
    run_command(export_cmd, "Step 1: Export slices from NIfTI")

    # Step 2: Run inference
    infer_cmd = [
        "python", "-m", "src.yolo.infer",
        "--model", str(args.model),
        "--images", "data/inference/slices/images",
        "--output", "outputs/inference",
        "--meta", "data/inference/slices/meta",
        "--format", "both",
    ]
    run_command(infer_cmd, "Step 2: Run model inference")

    print("\n" + "=" * 70)
    print("Inference Pipeline Complete!")
    print("=" * 70)
    print(f"Slices: data/inference/slices/images/")
    print(f"Predictions (YOLO): outputs/inference/labels/")
    print(f"Predictions (JSON): outputs/inference/predictions/")
    print("=" * 70)