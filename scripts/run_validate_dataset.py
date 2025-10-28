"""
Script to validate YOLO dataset integrity.

This is a convenience wrapper around src.yolo.validate_dataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.yolo.validate_dataset import main as validate_main


if __name__ == "__main__":
    # Override sys.argv to pass arguments
    sys.argv = [
        "validate_dataset",
        "--config", "configs/yolo_sagittal.yaml",
    ]
    
    print("=" * 70)
    print("Running YOLO Dataset Validation")
    print("=" * 70)
    print(f"Config: {project_root / 'configs' / 'yolo_sagittal.yaml'}")
    print("=" * 70)
    print()
    
    # Run validation
    validate_main()