# Complete Inference Workflow

This guide walks you through the entire inference pipeline from start to finish using the new batch processing features.

## Prerequisites

1. Trained model (in `outputs/runs/train_*/weights/best.pt`)
2. NIfTI files to process (`.nii.gz` format)
3. Conda environment activated

## Step-by-Step Guide

### 1. Prepare Your Data

Place all NIfTI files in the inference directory:

```powershell
# Copy your files to the inference folder
Copy-Item "C:\path\to\patient001.nii.gz" data\inference\
Copy-Item "C:\path\to\patient002.nii.gz" data\inference\
Copy-Item "C:\path\to\john-smith.nii.gz" data\inference\
```

Your structure should look like:
```
data/inference/
├── patient001.nii.gz
├── patient002.nii.gz
└── john-smith.nii.gz
```

**Note:** Filenames can be anything - patient IDs will be auto-derived.

---

### 2. Run Batch Inference

Process all patients with a single command:

```powershell
python scripts/run_inference_test_patient.py
```

**What happens:**
1. ✅ Finds all `.nii.gz` files in `data/inference/`
2. ✅ Exports 2D sagittal slices for each patient
3. ✅ Runs YOLOv8 segmentation on all slices
4. ✅ Saves predictions in both YOLO and JSON formats

**Output:**
```
data/inference/slices/
├── images/
│   ├── patient001_slice0000.png
│   ├── patient001_slice0001.png
│   ├── patient002_slice0000.png
│   └── ...
└── meta/
    └── *.json (geometry metadata)

outputs/inference/
├── labels/
│   └── *.txt (YOLO format)
└── predictions/
    └── *.json (JSON with pixel coords)
```

---

### 3. Create QC Overlays

Generate visual overlays for quality control:

```powershell
python scripts/run_create_overlays.py
```

**Output:**
```
outputs/qc/
├── patient001_slice0000_overlay.png
├── patient001_slice0001_overlay.png
├── patient002_slice0000_overlay.png
└── ...
```

**Review:**
- Open overlays in Windows Photo Viewer
- Use arrow keys to flip through quickly
- Look for missing detections or false positives
- Check IVF left/right balance (color-coded)

---

### 4. Convert to 3D NIfTI Masks

Reconstruct 3D masks for all patients:

```powershell
python scripts/run_predictions_to_nifti.py
```

**Output:**
```
outputs/masks/
├── patient001_pred_mask.nii.gz
├── patient002_pred_mask.nii.gz
└── john-smith_pred_mask.nii.gz
```

**Label values:**
- 0 = Background
- 1 = Vertebra
- 2 = IVF
- 3 = Sacrum
- 4 = Spinal Canal

---

### 5. View in 3D Slicer

1. **Launch 3D Slicer**

2. **Load Original Image:**
   - File → Add Data
   - Select `data/inference/patient001.nii.gz`
   - Click OK

3. **Load Prediction Mask:**
   - File → Add Data
   - Select `outputs/masks/patient001_pred_mask.nii.gz`
   - Check "Show Options"
   - Under "Description", select "LabelMap Volume"
   - Click OK

4. **View Results:**
   - Adjust window/level in slice viewers
   - Use "Segmentations" module to visualize in 3D
   - Export segments for further analysis

---

## Single Patient Mode

If you only want to process one patient:

```powershell
# Just inference
python scripts/run_inference_test_patient.py --input data/inference/patient001.nii.gz

# Just QC (after inference)
python scripts/run_create_overlays.py

# Just 3D mask
python scripts/run_predictions_to_nifti.py --patient_id patient001
```

---

## Custom Paths

### Different input directory
```powershell
python scripts/run_inference_test_patient.py --input "C:\Medical\Scans\MRI\"
```

### Different output locations
```powershell
python scripts/run_inference_test_patient.py --output_slices data/custom/slices --output_predictions outputs/custom/predictions
```

### Specific model
```powershell
python scripts/run_inference_test_patient.py --model outputs/runs/train_20251029_181626/weights/best.pt
```

---

## Complete Pipeline (All Steps)

Run everything in sequence:

```powershell
# 1. Inference
python scripts/run_inference_test_patient.py

# 2. QC Overlays
python scripts/run_create_overlays.py

# 3. 3D Reconstruction
python scripts/run_predictions_to_nifti.py
```

**Total time (example):**
- 3 patients × ~50 slices each
- Inference: ~2-3 minutes (GPU)
- QC Overlays: ~30 seconds
- 3D Reconstruction: ~10 seconds
- **Total: ~3-4 minutes for 3 patients**

---

## Tips & Tricks

### Process Large Batches
- Place 10+ NIfTI files in `data/inference/`
- Run batch inference once
- All downstream steps (QC, reconstruction) work automatically

### Organize by Date/Study
```
data/inference/
├── study_2025_01_patient001.nii.gz
├── study_2025_01_patient002.nii.gz
└── study_2025_02_patient003.nii.gz
```

### Quick QC Workflow
1. Run inference on all patients
2. Generate QC overlays
3. Open `outputs/qc/` in File Explorer
4. Sort by patient name
5. Press F11 for slideshow in Windows Photo Viewer

### Selective 3D Reconstruction
```powershell
# Only convert patients that passed QC
python scripts/run_predictions_to_nifti.py --patient_id patient001
python scripts/run_predictions_to_nifti.py --patient_id patient003
```

---

## Troubleshooting

### No GPU detected
- Inference will run on CPU (slower)
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

### Out of memory
- Process fewer patients at once
- Reduce batch size in model (requires re-training)

### Predictions don't align in 3D Slicer
- Ensure original image is provided for reconstruction
- Check that files haven't been resampled externally

### Missing predictions
- Check model confidence threshold (default: 0.25)
- Lower threshold: `python -m src.yolo.infer --conf 0.15 ...`

---

## Next Steps

- Quantitative analysis of segmented structures
- Export measurements (volume, shape)
- Clinical reporting
- Dataset expansion and retraining
