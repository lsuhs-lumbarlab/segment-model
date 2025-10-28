# Step 9: QC Overlay Visualization

## Overview

The `src/common/viz_overlay.py` script creates quality control visualizations by drawing predicted polygons on original images with class-specific colors.

## Quick Start

### Create overlays
```powershell
python scripts\run_create_overlays.py
```

## Features

- **Class-specific colors** for easy identification
- **Semi-transparent fills** to see underlying anatomy
- **Confidence scores** displayed on predictions
- **Pseudo-coloring for IVF** (left vs right differentiation for QC)

## Color Legend

| Class | Color | RGB |
|-------|-------|-----|
| Vertebra | 🟢 Green | (0, 255, 0) |
| IVF | 🔵 Blue | (255, 0, 0) |
| Sacrum | 🟠 Orange | (0, 165, 255) |
| Spinal Canal | 🟡 Yellow | (0, 255, 255) |

### IVF Pseudo-Coloring (QC Mode)

When `--pseudo-color-ivf` is enabled:
- **Left IVFs** (x < image_width/2) = Light blue
- **Right IVFs** (x > image_width/2) = Light red

This helps verify that the model is detecting IVFs on both sides.

## Custom Usage

### Different prediction format
```powershell
# Use YOLO format labels instead of JSON
python -m src.common.viz_overlay --images data/inference/slices/images --predictions outputs/inference/labels --output outputs/qc --format yolo
```

### Adjust visualization
```powershell
# Thicker lines, more transparent fills
python -m src.common.viz_overlay --images data/inference/slices/images --predictions outputs/inference/predictions --output outputs/qc --thickness 3 --alpha 0.2
```

### Training set overlays
```powershell
# Visualize ground truth labels
python -m src.common.viz_overlay --images data/derivatives/yolo_sagittal/images/train --predictions data/derivatives/yolo_sagittal/labels/train --output outputs/qc/train_gt --format yolo
```

## Output Structure
```
outputs/qc/
├─ patient_test001_slice0000_overlay.png
├─ patient_test001_slice0001_overlay.png
├─ patient_test001_slice0002_overlay.png
└─ ...
```

## Use Cases

1. **Verify model predictions** - Quick visual check of inference quality
2. **Compare to ground truth** - Overlay training labels for comparison
3. **Identify failure cases** - Spot missing or incorrect detections
4. **Present results** - Show stakeholders model performance

## Tips

- Open multiple overlays in Windows Photo Viewer and use arrow keys to quickly flip through
- Look for:
  - Missing detections (anatomy present but not detected)
  - False positives (detections where no anatomy exists)
  - Polygon quality (smooth vs jagged boundaries)
  - Left/right IVF balance

## Next Steps

Step 10 will create ROI crop generators for vertebra and IVF regions.