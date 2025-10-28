"""
Script to export sagittal slices from NIfTI volumes for YOLO training.

This is a convenience wrapper around src.yolo.export_slices
that uses hardcoded paths and settings for the project.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.yolo.export_slices import main as export_main


if __name__ == "__main__":
    # Override sys.argv to pass arguments to the export script
    sys.argv = [
        "export_slices",
        "--input_dir", "data/raw",
        "--output_dir", "data/derivatives/yolo_sagittal",
        "--train_ratio", "1.0",
        "--val_ratio", "0.0",
        "--test_ratio", "0.0",
        "--axis", "0",  # Sagittal
        "--seed", "42",
    ]
    
    print("=" * 70)
    print("Running Slice Export for YOLO Training")
    print("=" * 70)
    print(f"Input:  {project_root / 'data' / 'raw'}")
    print(f"Output: {project_root / 'data' / 'derivatives' / 'yolo_sagittal'}")
    print(f"Split:  100% training (all 6 patients)")
    print("=" * 70)
    print()
    
    # Run the export
    export_main()