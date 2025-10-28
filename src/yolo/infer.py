"""
Run YOLOv8 segmentation inference on spine MRI slices.

This script:
1. Loads a trained YOLOv8-seg model
2. Runs inference on test images
3. Saves predictions as YOLO polygon format
4. Denormalizes coordinates to original pixel space
5. Optionally uses SAHI tiling for large images
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


def load_geometry_metadata(json_path: Path) -> dict:
    """Load geometry metadata from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def denormalize_polygon(
    normalized_coords: List[float],
    image_width: int,
    image_height: int,
) -> List[float]:
    """
    Convert normalized [0, 1] coordinates to pixel coordinates.

    Args:
        normalized_coords: Flattened list [x1, y1, x2, y2, ...]
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Pixel coordinates [x1, y1, x2, y2, ...]
    """
    pixel_coords = []
    for i in range(0, len(normalized_coords), 2):
        x_norm = normalized_coords[i]
        y_norm = normalized_coords[i + 1]
        x_pixel = x_norm * image_width
        y_pixel = y_norm * image_height
        pixel_coords.extend([x_pixel, y_pixel])
    return pixel_coords


def run_inference(
    model: YOLO,
    image_path: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    device: str = None,
) -> List[Dict]:
    """
    Run inference on a single image.

    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        conf_threshold: Confidence threshold (default: 0.25)
        iou_threshold: IoU threshold for NMS (default: 0.7)
        device: Device to use (default: auto-detect)

    Returns:
        List of detection dictionaries
    """
    # Run prediction
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=False,
    )

    detections = []

    if results and len(results) > 0:
        result = results[0]

        # Check if masks exist
        if result.masks is None:
            return detections

        # Get image dimensions
        img_height, img_width = result.orig_shape

        # Process each detection
        for i in range(len(result.boxes)):
            box = result.boxes[i]
            mask = result.masks[i]

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            # Get mask polygon (already in normalized coords from YOLO)
            if hasattr(mask, 'xy') and len(mask.xy) > 0:
                # mask.xy gives list of polygons (for this mask)
                polygon_list = mask.xy[0]  # Get first polygon
                
                # Flatten and normalize
                normalized_coords = []
                for point in polygon_list:
                    x_norm = float(point[0]) / img_width
                    y_norm = float(point[1]) / img_height
                    normalized_coords.extend([x_norm, y_norm])

                detection = {
                    "class_id": class_id,
                    "confidence": confidence,
                    "normalized_polygon": normalized_coords,
                    "pixel_polygon": denormalize_polygon(
                        normalized_coords, img_width, img_height
                    ),
                }
                detections.append(detection)

    return detections


def save_predictions(
    detections: List[Dict],
    output_path: Path,
    geometry_metadata: Optional[dict] = None,
    format: str = "yolo",
):
    """
    Save predictions to file.

    Args:
        detections: List of detection dictionaries
        output_path: Output file path
        geometry_metadata: Optional geometry metadata to include
        format: Output format ("yolo" or "json")
    """
    if format == "yolo":
        # Save in YOLO format (normalized coordinates)
        lines = []
        for det in detections:
            coords_str = " ".join([f"{c:.6f}" for c in det["normalized_polygon"]])
            line = f"{det['class_id']} {coords_str}"
            lines.append(line)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    elif format == "json":
        # Save as JSON with pixel coordinates and metadata
        output_data = {
            "detections": [
                {
                    "class_id": det["class_id"],
                    "confidence": det["confidence"],
                    "polygon_pixels": det["pixel_polygon"],
                }
                for det in detections
            ],
            "geometry": geometry_metadata,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)


def process_inference_batch(
    model_path: Path,
    images_dir: Path,
    output_dir: Path,
    meta_dir: Optional[Path] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    device: str = None,
    save_format: str = "both",
) -> Dict[str, int]:
    """
    Run inference on a batch of images.

    Args:
        model_path: Path to trained model weights (.pt)
        images_dir: Directory containing input images
        output_dir: Directory to save predictions
        meta_dir: Optional directory containing geometry JSON files
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        device: Device to use (default: auto-detect)
        save_format: "yolo", "json", or "both"

    Returns:
        Dictionary with processing statistics
    """
    # Auto-detect device if not specified
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))

    # Get all image files
    image_files = sorted(images_dir.glob("*.png"))

    if not image_files:
        print(f"⚠ No PNG images found in {images_dir}")
        return {"processed": 0, "detections": 0}

    print(f"Found {len(image_files)} images")
    print(f"Running inference on device: {device}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")

    # Create output directories
    if save_format in ["yolo", "both"]:
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

    if save_format in ["json", "both"]:
        predictions_dir = output_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

    total_detections = 0

    for image_file in tqdm(image_files, desc="Processing images"):
        slice_name = image_file.stem

        # Run inference
        detections = run_inference(
            model, image_file, conf_threshold, iou_threshold, device
        )

        total_detections += len(detections)

        # Load geometry metadata if available
        geometry = None
        if meta_dir:
            meta_path = meta_dir / f"{slice_name}.json"
            if meta_path.exists():
                geometry = load_geometry_metadata(meta_path)

        # Save predictions
        if save_format in ["yolo", "both"]:
            label_path = labels_dir / f"{slice_name}.txt"
            save_predictions(detections, label_path, geometry, format="yolo")

        if save_format in ["json", "both"]:
            json_path = predictions_dir / f"{slice_name}.json"
            save_predictions(detections, json_path, geometry, format="json")

    return {"processed": len(image_files), "detections": total_detections}


def main():
    """CLI entry point for inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run YOLOv8 segmentation inference on spine MRI slices"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model weights (.pt file)",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/inference"),
        help="Output directory (default: outputs/inference)",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Directory containing geometry JSON files (optional)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS (default: 0.7)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto-detect, use '0' for GPU, 'cpu' for CPU)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["yolo", "json", "both"],
        help="Output format (default: both)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("YOLOv8 Segmentation Inference")
    print("=" * 70)

    # Verify inputs
    if not args.model.exists():
        print(f"✗ ERROR: Model not found: {args.model}")
        sys.exit(1)

    if not args.images.exists():
        print(f"✗ ERROR: Images directory not found: {args.images}")
        sys.exit(1)

    # Run inference
    try:
        stats = process_inference_batch(
            model_path=args.model,
            images_dir=args.images,
            output_dir=args.output,
            meta_dir=args.meta,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            save_format=args.format,
        )

        print("\n" + "=" * 70)
        print("Inference Complete")
        print("=" * 70)
        print(f"Images processed: {stats['processed']}")
        print(f"Total detections: {stats['detections']}")
        print(f"Output directory: {args.output.absolute()}")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Inference failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()