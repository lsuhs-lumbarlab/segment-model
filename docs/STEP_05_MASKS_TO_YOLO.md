# Step 5: Mask → YOLO Polygon Conversion

## Overview

The `src/yolo/masks_to_yolo.py` script converts binary mask PNGs to YOLO segmentation format (polygon annotations).

## Features

- **Connected component detection** per class
- **OpenCV contour extraction** with area filtering
- **Polygon simplification** using Douglas-Peucker algorithm
- **Normalized coordinates** [0, 1] for YOLO format
- **Multiple instances** per class automatically detected

## Class Mapping

Binary masks are converted using this mapping:

| Mask Class | YOLO Class ID | Name          |
|------------|---------------|---------------|
| 1          | 0             | vertebra      |
| 2          | 1             | ivf           |
| 3          | 2             | sacrum        |
| 4          | 3             | spinal_canal  |

## Usage

### Basic usage
```powershell
python scripts\run_masks_to_yolo.py
```

### Custom parameters
```powershell
python -m src.yolo.masks_to_yolo --data_dir data\derivatives\yolo_sagittal --splits train val test --min_area 50 --epsilon_factor 0.001
```

### Parameters

- `--data_dir`: Root data directory (default: `data/derivatives/yolo_sagittal`)
- `--splits`: Dataset splits to process (default: `train val test`)
- `--min_area`: Minimum contour area in pixels (default: 50)
- `--epsilon_factor`: Polygon simplification factor (default: 0.001, lower = more detailed)

## Output Format

YOLO segmentation format (one `.txt` file per image):
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Where:
- `class_id`: Integer class ID (0-3)
- `x1 y1 ... xn yn`: Normalized polygon coordinates [0, 1]
- One line per instance (multiple vertebrae = multiple lines with class 0)

## Output Structure
```
data/derivatives/yolo_sagittal/
├─ images/
│  └─ train/
│     └─ patient001_slice0005.png
├─ masks/        # Binary masks (input)
│  └─ train/
│     ├─ patient001_slice0005_class1.png
│     ├─ patient001_slice0005_class2.png
│     └─ ...
└─ labels/       # YOLO polygons (output)
   └─ train/
      └─ patient001_slice0005.txt
```

## Next Steps

Step 6 will create the YOLO dataset configuration file (data.yaml).