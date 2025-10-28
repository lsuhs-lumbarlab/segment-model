"""
Script to create QC overlay visualizations.

This is a convenience wrapper around src.common.viz_overlay.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.viz_overlay import main as viz_main


if __name__ == "__main__":
    # Override sys.argv to pass arguments
    sys.argv = [
        "viz_overlay",
        "--images", "data/inference/slices/images",
        "--predictions", "outputs/inference/predictions",
        "--output", "outputs/qc",
        "--format", "json",
        "--pseudo-color-ivf",  # Enable IVF left/right coloring
    ]
    
    print("=" * 70)
    print("Running QC Overlay Creation")
    print("=" * 70)
    print(f"Images: {project_root / 'data' / 'inference' / 'slices' / 'images'}")
    print(f"Predictions: {project_root / 'outputs' / 'inference' / 'predictions'}")
    print(f"Output: {project_root / 'outputs' / 'qc'}")
    print(f"Pseudo-color IVF: Enabled (left=light blue, right=light red)")
    print("=" * 70)
    print()
    
    # Run visualization
    viz_main()