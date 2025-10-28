"""
Script to train YOLOv8 segmentation model.

This is a convenience wrapper around src.yolo.train.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.yolo.train import main as train_main


if __name__ == "__main__":
    # Override sys.argv to pass arguments
    sys.argv = [
        "train",
        "--data", "configs/yolo_sagittal.yaml",
        "--model", "yolov8m-seg.pt",
        "--epochs", "100",
        "--imgsz", "640",
        "--batch", "8",
        "--patience", "50",
    ]
    
    print("=" * 70)
    print("Running YOLOv8 Segmentation Training")
    print("=" * 70)
    print(f"Config: {project_root / 'configs' / 'yolo_sagittal.yaml'}")
    print(f"Model: yolov8m-seg (medium)")
    print(f"Epochs: 100 (with early stopping patience=50)")
    print(f"Image size: 640x640")
    print(f"Batch size: 8")
    print("=" * 70)
    print()
    
    # Run training
    train_main()