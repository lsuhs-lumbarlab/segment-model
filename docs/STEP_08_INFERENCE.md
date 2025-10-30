# Step 8: Inference

## Updated Folder Structure

The inference pipeline now supports a simplified flat folder structure:

```
data/
└── inference/
    ├── patient_test001.nii.gz
    ├── patient_test002.nii.gz
    ├── john-smith.nii.gz
    └── any-filename.nii.gz
```

**No more nested folders required!** Just place all your NIfTI files directly in `data/inference/`.

---

## Usage Examples

### 1. Batch Mode (Recommended) - Process All Files

Process all `.nii.gz` files in the default `data/inference/` directory:

```powershell
python scripts/run_inference_test_patient.py
```

This will:
- ✅ Auto-detect all `.nii.gz` files in `data/inference/`
- ✅ Auto-derive patient IDs from filenames (e.g., `john-smith.nii.gz` → `john-smith`)
- ✅ Process all patients in one command
- ✅ Auto-detect the latest trained model

---

### 2. Batch Mode - Custom Directory

Process all `.nii.gz` files in a custom directory:

```powershell
python scripts/run_inference_test_patient.py --input "C:/path/to/nifti/files/"
```

---

### 3. Single File Mode

Process a single NIfTI file:

```powershell
# Auto-derive patient ID from filename
python scripts/run_inference_test_patient.py --input data/inference/patient_test001.nii.gz

# Or specify a custom patient ID
python scripts/run_inference_test_patient.py --input data/inference/john-smith.nii.gz --patient_id john_smith
```

---

### 4. Custom Model and Output Paths

```powershell
python scripts/run_inference_test_patient.py \
    --input data/inference/ \
    --model outputs/runs/train_20251029_181626/weights/best.pt \
    --output_slices data/inference/slices \
    --output_predictions outputs/inference
```

---

## Output Structure

After running inference, you'll find:

```
data/
└── inference/
    └── slices/
        ├── images/
        │   ├── patient_test001_slice0000.png
        │   ├── patient_test001_slice0001.png
        │   ├── john-smith_slice0000.png
        │   └── ...
        └── meta/
            ├── patient_test001_slice0000.json
            ├── patient_test001_slice0001.json
            └── ...

outputs/
└── inference/
    ├── labels/                      # YOLO format predictions
    │   ├── patient_test001_slice0000.txt
    │   ├── john-smith_slice0000.txt
    │   └── ...
    └── predictions/                 # JSON format with metadata
        ├── patient_test001_slice0000.json
        ├── john-smith_slice0000.json
        └── ...
```

---

## Advanced: Module-Level Usage

You can also call the modules directly:

### Export slices only:
```powershell
# Batch mode (default: data/inference/)
python -m src.yolo.export_test_slices

# Custom directory
python -m src.yolo.export_test_slices --input path/to/nifti/files/

# Single file
python -m src.yolo.export_test_slices --input patient001.nii.gz
```

### Run inference on pre-exported slices:
```powershell
python -m src.yolo.infer \
    --model outputs/runs/train_20251029_181626/weights/best.pt \
    --images data/inference/slices/images \
    --output outputs/inference \
    --meta data/inference/slices/meta
```

---

## Migration from Old Structure

**Old structure** (nested folders):
```
data/inference/
├── patient_test001/
│   └── image.nii.gz
└── patient_test002/
    └── image.nii.gz
```

**New structure** (flat):
```
data/inference/
├── patient_test001.nii.gz
└── patient_test002.nii.gz
```

Simply rename and move files:
```powershell
# PowerShell
Move-Item data/inference/patient_test001/image.nii.gz data/inference/patient_test001.nii.gz
Move-Item data/inference/patient_test002/image.nii.gz data/inference/patient_test002.nii.gz

# Remove empty directories
Remove-Item data/inference/patient_test001 -Recurse
Remove-Item data/inference/patient_test002 -Recurse
```

---

## Notes

- **Filenames**: Can be anything (e.g., `patient_test001.nii.gz`, `john-smith.nii.gz`, `MRI_2024_001.nii.gz`)
- **Patient IDs**: Auto-derived from filename by removing `.nii.gz` extension
- **Model**: Auto-detects latest trained model from `outputs/runs/` if not specified
- **Backward Compatibility**: Single-file mode still works with `--patient_id` argument
