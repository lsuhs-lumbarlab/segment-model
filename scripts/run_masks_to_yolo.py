"""
Script to convert binary masks to YOLO polygon format.

This is a convenience wrapper around src.yolo.masks_to_yolo.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.yolo.masks_to_yolo import main as masks_to_yolo_main


if __name__ == "__main__":
    # Override sys.argv to pass arguments
    sys.argv = [
        "masks_to_yolo",
        "--data_dir", "data/derivatives/yolo_sagittal",
        "--splits", "train",  # Only train for now since val/test are empty
        "--min_area", "50",
        "--epsilon_factor", "0.001",
    ]
    
    print("=" * 70)
    print("Running Mask to YOLO Polygon Conversion")
    print("=" * 70)
    print(f"Input:  {project_root / 'data' / 'derivatives' / 'yolo_sagittal' / 'masks'}")
    print(f"Output: {project_root / 'data' / 'derivatives' / 'yolo_sagittal' / 'labels'}")
    print(f"Processing: train split only")
    print("=" * 70)
    print()
    
    # Run the conversion
    masks_to_yolo_main()