"""
Script to create QC overlay visualizations.

This script creates visual overlays of predictions on images for quality control.
Works automatically with batch inference output.

Usage:
    # Default - use output from batch inference
    python scripts/run_create_overlays.py
    
    # Custom paths
    python scripts/run_create_overlays.py --images data/custom/images --predictions outputs/custom/predictions
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.viz_overlay import create_qc_overlays


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create QC overlay visualizations for inference results"
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/inference/slices/images"),
        help="Directory containing slice images (default: data/inference/slices/images)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/inference/predictions"),
        help="Directory containing predictions (default: outputs/inference/predictions)",
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
        "--no-pseudo-color",
        action="store_true",
        help="Disable IVF left/right pseudo-coloring",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("QC Overlay Creation")
    print("=" * 70)
    print(f"Images: {args.images.absolute()}")
    print(f"Predictions: {args.predictions.absolute()}")
    print(f"Output: {args.output.absolute()}")
    print(f"Format: {args.format}")
    print(f"Pseudo-color IVF: {'Disabled' if args.no_pseudo_color else 'Enabled (left=light blue, right=light red)'}")
    print("=" * 70)
    
    # Verify inputs
    if not args.images.exists():
        print(f"\n✗ ERROR: Images directory not found: {args.images}")
        print("Have you run inference yet? Try: python scripts/run_inference_test_patient.py")
        sys.exit(1)
    
    if not args.predictions.exists():
        print(f"\n✗ ERROR: Predictions directory not found: {args.predictions}")
        print("Have you run inference yet? Try: python scripts/run_inference_test_patient.py")
        sys.exit(1)
    
    # Create overlays
    try:
        create_qc_overlays(
            images_dir=args.images,
            predictions_dir=args.predictions,
            output_dir=args.output,
            prediction_format=args.format,
            pseudo_color_ivf=not args.no_pseudo_color,
        )
        
        print("\n" + "=" * 70)
        print("QC Overlays Complete!")
        print("=" * 70)
        print(f"Overlays saved to: {args.output.absolute()}")
        print("\nReview the overlay images to verify prediction quality.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Failed to create overlays: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)