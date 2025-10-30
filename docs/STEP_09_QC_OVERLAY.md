# Step 9: QC Overlay Visualization

## Overview

The QC overlay script creates quality control visualizations by drawing predicted polygons on original images with class-specific colors. This works automatically with batch inference output.

## Quick Start

### Create overlays for all inference results
```powershell
python scripts/run_create_overlays.py
```

That's it! This will process all slices from your batch inference and create overlay images in `outputs/qc/`.

## Custom Usage

### Custom paths
```powershell
python scripts/run_create_overlays.py --images data/custom/images --predictions outputs/custom/predictions --output outputs/qc_custom
```

### Different prediction format
```powershell
# Use YOLO format labels instead of JSON
python scripts/run_create_overlays.py --format yolo
```

### Disable IVF pseudo-coloring
```powershell
python scripts/run_create_overlays.py --no-pseudo-color
```

### Training set overlays
```powershell
# Visualize ground truth labels
python -m src.common.viz_overlay --images data/derivatives/yolo_sagittal/images/train --predictions data/derivatives/yolo_sagittal/labels/train --output outputs/qc/train_gt --format yolo
```

## Features

- **Batch processing** - Automatically handles all inference results
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

When enabled (default):
- **Left IVFs** (x < image_width/2) = Light blue
- **Right IVFs** (x > image_width/2) = Light red

This helps verify that the model is detecting IVFs on both sides.

## Output Structure
```
outputs/qc/
├─ patient_test001_slice0000_overlay.png
├─ patient_test001_slice0001_overlay.png
├─ patient_test002_slice0000_overlay.png
├─ john-smith_slice0000_overlay.png
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

- [Convert predictions to 3D NIfTI masks](STEP_10_NIFTI_RECONSTRUCTION.md) for viewing in 3D Slicer