# Step 10: Convert Predictions to 3D NIfTI Masks

## Overview

After running inference on 2D slices, you can reconstruct the predictions back into 3D NIfTI masks for viewing in 3D Slicer or other medical imaging software.

## Quick Start

### Batch Mode - Convert All Patients

```powershell
python scripts/run_predictions_to_nifti.py
```

This will:
- ✅ Auto-detect all patients from prediction files
- ✅ Auto-find original NIfTI images (for exact affine/header)
- ✅ Reconstruct 3D masks for all patients
- ✅ Save masks to `outputs/masks/`

### Single Patient

```powershell
python scripts/run_predictions_to_nifti.py --patient_id patient_test001
```

### With Specific Original Image

```powershell
python scripts/run_predictions_to_nifti.py --patient_id patient_test001 --original_image data/inference/patient_test001.nii.gz
```

## Output

### File Structure
```
outputs/
└── masks/
    ├── patient_test001_pred_mask.nii.gz
    ├── patient_test002_pred_mask.nii.gz
    ├── john-smith_pred_mask.nii.gz
    └── ...
```

### Label Values

The reconstructed masks use the following integer labels:

| Value | Anatomical Structure |
|-------|---------------------|
| 0 | Background |
| 1 | Vertebra |
| 2 | IVF (Intervertebral Foramen) |
| 3 | Sacrum |
| 4 | Spinal Canal |

## Viewing in 3D Slicer

1. **Load Original Image:**
   - File → Add Data → Select `patient_test001.nii.gz`

2. **Load Prediction Mask:**
   - File → Add Data → Select `patient_test001_pred_mask.nii.gz`
   - Make sure to load it as "LabelMap Volume"

3. **Set Up Visualization:**
   - In the Volumes module, set the mask as a label map
   - Adjust color table if needed
   - Use the "Editor" module to view labels

4. **3D Visualization:**
   - Switch to "Volume Rendering" module
   - Apply preset for the original image
   - In "Segmentations" module, create segmentation from labelmap
   - Show 3D to see segmented structures

## Advanced Usage

### Custom Paths

```powershell
python scripts/run_predictions_to_nifti.py --predictions outputs/inference/predictions --output outputs/custom_masks --inference_dir data/inference
```

### Module-Level Usage

```powershell
python -m src.yolo.predictions_to_nifti --predictions outputs/inference/predictions --patient_id patient_test001 --output outputs/masks/patient_test001_pred_mask.nii.gz --original_image data/inference/patient_test001.nii.gz
```

## How It Works

The reconstruction process:

1. **Find Prediction Files:** Collects all `{patient_id}_slice*.json` files
2. **Load Geometry:** Reads spacing, affine, and slice indices from metadata
3. **Convert Polygons to Masks:** Draws each polygon as filled regions with class labels
4. **Stack Slices:** Reconstructs 3D volume in RAS orientation
5. **Apply Affine:** Uses original image affine or metadata affine
6. **Save NIfTI:** Writes as int16 NIfTI with proper header

## Coordinate Space

- **Input predictions:** 2D pixel coordinates per slice
- **Reconstruction:** 3D RAS (Right-Anterior-Superior) orientation
- **Output mask:** Same coordinate space as original image

This ensures the mask overlays perfectly with the original image in 3D Slicer.

## Troubleshooting

### Mask doesn't align with image

Make sure you're providing the original image:
```powershell
python scripts/run_predictions_to_nifti.py --patient_id patient_test001 --original_image data/inference/patient_test001.nii.gz
```

### Missing slices

Check that all prediction JSON files exist:
```powershell
ls outputs/inference/predictions/patient_test001_slice*.json
```

### Wrong orientation

The script automatically handles rotation applied during slice export. If you modified the export pipeline, ensure the rotation is properly reversed.

## Quality Control

After reconstruction:
1. **Visual inspection** in 3D Slicer - ensure mask aligns with anatomy
2. **Check slice-by-slice** - use slice viewers to verify predictions
3. **3D rendering** - visualize segmented structures in 3D
4. **Compare to QC overlays** - correlate 3D mask with 2D overlay images

## Next Steps

- Use masks for quantitative analysis (volume, shape metrics)
- Export masks for clinical review
- Create reports with measurements
- Further post-processing or refinement
