"""
Train YOLOv8 segmentation model on spine MRI dataset.

This script wraps Ultralytics YOLO with project-specific defaults
and handles GPU detection automatically.
"""

import sys
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO


def train_yolo_segmentation(
    data_config: Path,
    model: str = "yolov8m-seg.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    patience: int = 50,
    save_dir: Path = None,
    device: str = None,
    pretrained: bool = True,
    **kwargs,
):
    """
    Train YOLOv8 segmentation model.

    Args:
        data_config: Path to dataset YAML config
        model: Model architecture (default: yolov8m-seg.pt)
        epochs: Number of training epochs (default: 100)
        imgsz: Input image size (default: 640)
        batch: Batch size (default: 8, auto-adjusted if OOM)
        patience: Early stopping patience (default: 50)
        save_dir: Directory to save results (default: outputs/runs/train)
        device: Device to use (default: auto-detect)
        pretrained: Use pretrained weights (default: True)
        **kwargs: Additional arguments passed to YOLO.train()

    Returns:
        Training results
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "0"  # Use first GPU
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            print("⚠ No GPU detected, using CPU (training will be slow)")

    # Set default save directory
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(f"outputs/runs/train_{timestamp}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("YOLOv8 Segmentation Training")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Data config: {data_config}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print(f"Save directory: {save_dir.absolute()}")
    print(f"Pretrained: {pretrained}")
    print("=" * 70)

    # Load model
    if pretrained:
        print(f"\nLoading pretrained model: {model}")
        yolo = YOLO(model)
    else:
        print(f"\nInitializing model from scratch: {model}")
        # For training from scratch, use .yaml config
        model_yaml = model.replace("-seg.pt", "-seg.yaml")
        yolo = YOLO(model_yaml)

    # Train
    print("\nStarting training...\n")
    results = yolo.train(
        data=str(data_config),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        verbose=True,
        # Optimization settings
        optimizer="AdamW",
        lr0=0.001,  # Initial learning rate
        lrf=0.01,   # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # No rotation for medical images
        translate=0.1,
        scale=0.5,
        shear=0.0,  # No shear for medical images
        perspective=0.0,  # No perspective for medical images
        flipud=0.0,  # No vertical flip
        fliplr=0.5,  # Horizontal flip OK for sagittal
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # Validation
        val=True,
        plots=True,
        save=True,
        save_period=-1,  # Save only best and last
        # Additional kwargs
        **kwargs,
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Results saved to: {save_dir.absolute()}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last weights: {save_dir / 'weights' / 'last.pt'}")
    print("=" * 70)

    return results


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train YOLOv8 segmentation model on spine MRI dataset"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/yolo_sagittal.yaml"),
        help="Path to dataset YAML config (default: configs/yolo_sagittal.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m-seg.pt",
        choices=["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"],
        help="Model size (default: yolov8m-seg.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8, -1 for auto)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto-detect, use '0' for GPU, 'cpu' for CPU)",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Save directory (default: outputs/runs/train_TIMESTAMP)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch (no pretrained weights)",
    )

    args = parser.parse_args()

    # Verify data config exists
    if not args.data.exists():
        print(f"✗ ERROR: Data config not found: {args.data}")
        sys.exit(1)

    # Train
    try:
        train_yolo_segmentation(
            data_config=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            save_dir=args.save_dir,
            device=args.device,
            pretrained=not args.no_pretrained,
        )
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()