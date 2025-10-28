# Step 3: NIfTI Loader & Harmonizer

## Overview

The `src/common/io_nifti.py` module provides utilities for loading NIfTI medical images and ensuring spatial alignment between image and label pairs.

## Features

- **Automatic reorientation** to RAS (or custom orientation)
- **Spatial alignment verification**:
  - Shape matching
  - Spacing (voxel size) matching
  - Affine matrix matching
- **CLI tool** for quick validation

## Usage

### CLI: Check image/label alignment
```powershell
python -m src.common.io_nifti --check "path\to\image.nii.gz" "path\to\label.nii.gz"
```

### Python API
```python
from pathlib import Path
from src.common.io_nifti import load_and_reorient_nifti, verify_spatial_alignment

# Load and reorient to RAS
img_data, img_nib = load_and_reorient_nifti(Path("image.nii.gz"))
label_data, label_nib = load_and_reorient_nifti(Path("label.nii.gz"))

# Verify alignment
aligned = verify_spatial_alignment(img_nib, label_nib)
```

## Output

The tool prints:
- Current orientation
- Reorientation status
- Shape, spacing, and affine verification
- Exit code: 0 (pass) or 1 (fail)

## Next Steps

This module will be used by the slice exporter (Step 4) to ensure all image/label pairs are properly aligned before extracting 2D slices.