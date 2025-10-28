# Step 6: Dataset Configuration

## Overview

The YOLO dataset configuration file (`configs/yolo_sagittal.yaml`) defines the dataset structure, paths, and class mappings for training.

## Configuration File

**Location:** `configs/yolo_sagittal.yaml`

### Structure
```yaml
path: ../data/derivatives/yolo_sagittal  # Root dataset directory
train: images/train
val: images/val
test: images/test

names:
  0: vertebra
  1: ivf
  2: sacrum
  3: spinal_canal

nc: 4  # Number of classes
```

### Class Definitions

| Class ID | Name          | Description                    |
|----------|---------------|--------------------------------|
| 0        | vertebra      | Vertebral bodies               |
| 1        | ivf           | Intervertebral foramen         |
| 2        | sacrum        | Sacrum                         |
| 3        | spinal_canal  | Spinal canal                   |

## Dataset Validation

### Validate dataset integrity
```powershell
python scripts\run_validate_dataset.py
```

### What it checks

- ✓ Image/label count match
- ✓ Missing label files
- ✓ Missing image files
- ✓ Empty label files
- ✓ Class distribution per split

### Expected Output
```
TRAIN Split Validation
======================================================================
  Images: 29
  Labels: 29
  Matched: 29

  Class distribution:
    0 (vertebra): 45 instances
    1 (ivf): 38 instances
    2 (sacrum): 12 instances
    3 (spinal_canal): 29 instances

  ✓ TRAIN split is valid
```

## Troubleshooting

### Missing labels
- Check that `run_masks_to_yolo.py` completed successfully
- Verify masks exist in `data/derivatives/yolo_sagittal/masks/train/`

### Empty splits (val/test)
- Normal if you used `--train_ratio 1.0` during slice export
- Val/test will be added when you get more annotated data

## Next Steps

Step 7 will create the training script using this configuration.