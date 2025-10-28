# Step 4: Slice Exporter + Geometry JSON

## Overview

The `src/yolo/export_slices.py` script extracts 2D sagittal slices from 3D NIfTI volumes and prepares them for YOLO training.

## Features

- **RAS orientation**: Automatically reorients volumes to RAS before slicing
- **Robust normalization**: Uses 1st-99th percentile for intensity scaling
- **Patient-level splitting**: Train/val/test split by patient (not by slice)
- **Geometry metadata**: Saves spacing, origin, and affine matrix per slice
- **Multi-class masks**: Exports binary masks per class for polygon extraction

## Usage

### Basic usage (default 70/15/15 split)
```powershell
python -m src.yolo.export_slices --input_dir data\raw --output_dir data\derivatives\yolo_sagittal
```

### Custom split ratios
```powershell
python -m src.yolo.export_slices --input_dir data\raw --output_dir data\derivatives\yolo_sagittal --train_ratio 0.6 --val_ratio 0.2 --test_ratio 0.2
```

### Different slice axis
```powershell
# Coronal slices
python -m src.yolo.export_slices --axis 1

# Axial slices
python -m src.yolo.export_slices --axis 2
```

## Output Structure
```
data/derivatives/yolo_sagittal/
├─ images/
│  ├─ train/
│  │  └─ patient001_slice0000.png
│  ├─ val/
│  └─ test/
├─ masks/
│  ├─ train/
│  │  ├─ patient001_slice0000_class1.png  # vertebra
│  │  ├─ patient001_slice0000_class2.png  # ivf
│  │  └─ ...
│  ├─ val/
│  └─ test/
└─ meta/
   ├─ train/
   │  └─ patient001_slice0000.json  # Geometry metadata
   ├─ val/
   └─ test/
```

## Geometry JSON Format

Each slice has a corresponding JSON file with:
```json
{
  "patient_id": "patient001",
  "slice_index": 50,
  "axis": 0,
  "axis_name": "sagittal",
  "spacing_mm": [1.0, 1.0, 1.0],
  "origin_ras": [0.0, 0.0, 0.0],
  "affine": [[...], [...], [...], [...]],
  "image_shape": [512, 512],
  "labels_present": [1, 2, 3, 4]
}
```

## Next Steps

Step 5 will convert these binary masks to YOLO polygon format (.txt files).