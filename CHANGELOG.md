# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed - 2025-10-30
- **BREAKING**: Updated inference folder structure from nested to flat layout
  - OLD: `/data/inference/patient_test001/image.nii.gz`
  - NEW: `/data/inference/patient_test001.nii.gz`
- `export_test_slices.py`: Added batch processing mode to handle multiple NIfTI files
  - Now accepts directory input and processes all `.nii.gz` files
  - Auto-derives patient IDs from filenames
  - Maintains backward compatibility for single-file mode
- `run_inference_test_patient.py`: Enhanced to support batch inference
  - Default mode processes all files in `data/inference/`
  - No longer requires `--input` and `--patient_id` arguments (optional)
  - Auto-detects whether input is file or directory
  - Supports flexible filenames (e.g., `john-smith.nii.gz`, `MRI_2024_001.nii.gz`)
- `run_create_overlays.py`: Updated to be more user-friendly
  - Added command-line arguments for flexibility
  - Improved error messages with helpful suggestions
  - Better documentation and usage examples
- `run_predictions_to_nifti.py`: Complete rewrite to support batch processing
  - **NEW**: Batch mode processes all patients with single command
  - Auto-detects patient IDs from prediction files
  - Auto-finds original NIfTI images for exact affine/header
  - Provides detailed per-patient status reporting
  - Maintains backward compatibility for single-patient mode

### Added - 2025-10-30
- `docs/INFERENCE_USAGE.md`: Comprehensive guide for new inference workflow
- `docs/STEP_10_NIFTI_RECONSTRUCTION.md`: Complete guide for converting predictions to 3D masks
- README: Added "Quick Inference" section with simple usage examples
- Batch processing capability for inference pipeline
- Batch processing capability for NIfTI reconstruction
- Auto-detection of patient IDs from filenames
- Auto-detection of original images for reconstruction

### Fixed - 2025-10-30
- QC overlay script now has proper error messages when inference hasn't been run
- NIfTI reconstruction now properly handles finding original images in new flat structure

### Added - 2025-10-28
- Initial repository scaffold with Windows-friendly structure
- Base `.gitignore` for Python, data, and model artifacts
- README with Windows Quickstart guide
```

---

#### **5. `LICENSE`** (example - adjust as needed)

**Path:** `segment-model\LICENSE`
```
MIT License

Copyright (c) 2025 [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.