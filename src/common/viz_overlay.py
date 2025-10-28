"""
Visualization utilities for quality control overlays.

Draw predicted polygons on images with class-specific colors.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm


# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),      # vertebra - green
    1: (255, 0, 0),      # ivf - blue
    2: (0, 165, 255),    # sacrum - orange
    3: (0, 255, 255),    # spinal_canal - yellow
}

CLASS_NAMES = {
    0: "vertebra",
    1: "ivf",
    2: "sacrum",
    3: "spinal_canal",
}


def load_predictions_json(json_path: Path) -> List[Dict]:
    """Load predictions from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("detections", [])


def load_predictions_yolo(txt_path: Path, image_width: int, image_height: int) -> List[Dict]:
    """
    Load predictions from YOLO format TXT file.

    Args:
        txt_path: Path to YOLO label file
        image_width: Image width for denormalization
        image_height: Image height for denormalization

    Returns:
        List of detection dictionaries
    """
    detections = []

    if not txt_path.exists():
        return detections

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:  # At least class_id + 3 points (x,y pairs)
                continue

            class_id = int(parts[0])
            normalized_coords = [float(x) for x in parts[1:]]

            # Denormalize to pixel coordinates
            pixel_coords = []
            for i in range(0, len(normalized_coords), 2):
                x = normalized_coords[i] * image_width
                y = normalized_coords[i + 1] * image_height
                pixel_coords.extend([x, y])

            detections.append({
                "class_id": class_id,
                "confidence": 1.0,  # Not stored in YOLO format
                "polygon_pixels": pixel_coords,
            })

    return detections


def get_polygon_centroid(polygon: List[float]) -> Tuple[float, float]:
    """
    Calculate centroid of a polygon.

    Args:
        polygon: Flattened list of coordinates [x1, y1, x2, y2, ...]

    Returns:
        (centroid_x, centroid_y)
    """
    points = np.array(polygon).reshape(-1, 2)
    centroid_x = np.mean(points[:, 0])
    centroid_y = np.mean(points[:, 1])
    return centroid_x, centroid_y


def pseudo_color_ivf_by_side(
    detections: List[Dict],
    image_width: int,
) -> List[Dict]:
    """
    Pseudo-color IVF detections based on left/right position (for QC only).

    IVFs on the left side of image (x < width/2) get one color,
    IVFs on the right side get another color.

    Args:
        detections: List of detection dictionaries
        image_width: Image width in pixels

    Returns:
        Modified detections with updated colors
    """
    midline = image_width / 2

    for det in detections:
        if det["class_id"] == 1:  # IVF class
            polygon = det["polygon_pixels"]
            centroid_x, _ = get_polygon_centroid(polygon)

            # Assign pseudo-color based on side
            if centroid_x < midline:
                det["color"] = (255, 100, 100)  # Light blue (left)
            else:
                det["color"] = (100, 100, 255)  # Light red (right)

    return detections


def draw_polygon_overlay(
    image: np.ndarray,
    detections: List[Dict],
    thickness: int = 2,
    alpha: float = 0.3,
    show_labels: bool = True,
    pseudo_color_ivf: bool = False,
) -> np.ndarray:
    """
    Draw predicted polygons on image.

    Args:
        image: Input image (BGR)
        detections: List of detection dictionaries
        thickness: Line thickness (default: 2)
        alpha: Fill transparency (default: 0.3)
        show_labels: Show class labels (default: True)
        pseudo_color_ivf: Use pseudo-coloring for left/right IVF (default: False)

    Returns:
        Image with overlays
    """
    overlay = image.copy()
    output = image.copy()

    # Pseudo-color IVF if requested
    if pseudo_color_ivf:
        detections = pseudo_color_ivf_by_side(detections, image.shape[1])

    for det in detections:
        class_id = det["class_id"]
        polygon = det["polygon_pixels"]
        confidence = det.get("confidence", 1.0)

        # Get color
        if "color" in det:
            color = det["color"]
        else:
            color = CLASS_COLORS.get(class_id, (255, 255, 255))

        # Convert polygon to numpy array
        points = np.array(polygon).reshape(-1, 2).astype(np.int32)

        # Draw filled polygon on overlay
        cv2.fillPoly(overlay, [points], color)

        # Draw outline on output
        cv2.polylines(output, [points], isClosed=True, color=color, thickness=thickness)

        # Draw label
        if show_labels:
            centroid_x, centroid_y = get_polygon_centroid(polygon)
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            label = f"{class_name} {confidence:.2f}"

            # Put text with background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                output,
                (int(centroid_x) - 5, int(centroid_y) - text_h - 5),
                (int(centroid_x) + text_w + 5, int(centroid_y) + 5),
                color,
                -1,
            )
            cv2.putText(
                output,
                label,
                (int(centroid_x), int(centroid_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Blend overlay with output
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def create_qc_overlays(
    images_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    prediction_format: str = "json",
    pseudo_color_ivf: bool = False,
    thickness: int = 2,
    alpha: float = 0.3,
):
    """
    Create QC overlay images for all predictions.

    Args:
        images_dir: Directory containing original images
        predictions_dir: Directory containing predictions (JSON or TXT)
        output_dir: Directory to save overlay images
        prediction_format: "json" or "yolo"
        pseudo_color_ivf: Use pseudo-coloring for IVF left/right
        thickness: Line thickness
        alpha: Fill transparency
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = sorted(images_dir.glob("*.png"))

    if not image_files:
        print(f"⚠ No images found in {images_dir}")
        return

    print(f"Creating overlays for {len(image_files)} images...")

    for image_file in tqdm(image_files, desc="Creating overlays"):
        slice_name = image_file.stem

        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  ⚠ Failed to load image: {image_file}")
            continue

        # Load predictions
        if prediction_format == "json":
            pred_path = predictions_dir / f"{slice_name}.json"
            if pred_path.exists():
                detections = load_predictions_json(pred_path)
            else:
                detections = []
        else:  # yolo
            pred_path = predictions_dir / f"{slice_name}.txt"
            h, w = image.shape[:2]
            detections = load_predictions_yolo(pred_path, w, h)

        # Draw overlays
        overlay_image = draw_polygon_overlay(
            image,
            detections,
            thickness=thickness,
            alpha=alpha,
            show_labels=True,
            pseudo_color_ivf=pseudo_color_ivf,
        )

        # Save
        output_path = output_dir / f"{slice_name}_overlay.png"
        cv2.imwrite(str(output_path), overlay_image)

    print(f"✓ Overlays saved to: {output_dir}")


def main():
    """CLI entry point for creating QC overlays."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create QC overlay visualizations"
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing original images",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Directory containing predictions (JSON or TXT)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/qc"),
        help="Output directory for overlays (default: outputs/qc)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "yolo"],
        help="Prediction format (default: json)",
    )
    parser.add_argument(
        "--pseudo-color-ivf",
        action="store_true",
        help="Pseudo-color IVF by left/right position (QC only)",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Line thickness (default: 2)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Fill transparency (default: 0.3)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("QC Overlay Visualization")
    print("=" * 70)

    # Verify inputs
    if not args.images.exists():
        print(f"✗ ERROR: Images directory not found: {args.images}")
        sys.exit(1)

    if not args.predictions.exists():
        print(f"✗ ERROR: Predictions directory not found: {args.predictions}")
        sys.exit(1)

    # Create overlays
    try:
        create_qc_overlays(
            images_dir=args.images,
            predictions_dir=args.predictions,
            output_dir=args.output,
            prediction_format=args.format,
            pseudo_color_ivf=args.pseudo_color_ivf,
            thickness=args.thickness,
            alpha=args.alpha,
        )

        print("\n" + "=" * 70)
        print("QC Overlays Complete")
        print("=" * 70)
        print(f"Output directory: {args.output.absolute()}")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Failed to create overlays: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()