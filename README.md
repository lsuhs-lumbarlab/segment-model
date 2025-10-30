# AI Spine Segmentation (YOLOv8-seg)

Windows-based pipeline for training YOLOv8 instance segmentation on MRI spine data (sagittal slices).

## Project Status

🚧 **Phase 1: 2D Sagittal Segmentation** (In Progress)

**Classes:**
- `vertebra` (class 0)
- `ivf` (intervertebral foramen, class 1)
- `sacrum` (class 2)
- `spinal_canal` (class 3)

---

## Repository Structure
```
segment-model/
├─ src/
│  ├─ common/        # Shared utilities (IO, geometry, visualization)
│  └─ yolo/          # YOLO data prep, training, inference
├─ configs/          # YAML configs for datasets and models
├─ data/             # Raw and processed data (gitignored)
├─ outputs/          # Model outputs and QC results (gitignored)
├─ docs/             # Detailed documentation
├─ scripts/          # Windows batch scripts (.bat)
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ environment.yml
└─ CHANGELOG.md
```

---

## Windows Quickstart

### Prerequisites
- Windows 10/11
- NVIDIA GPU (CUDA capable)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed
- Git for Windows
- Visual Studio Code (recommended)

### Setup

1. **Clone the repository:**
```powershell
   git clone https://github.com/lsuhs-lumbarlab/segment-model
   cd segment-model
```

2. **Activate your existing conda environment:**
```powershell
   conda activate segment-model
```

3. **Install dependencies**

   **Option A: Use existing environment (if you already have segment-model):**
```powershell
   conda activate segment-model
   
   # Install PyTorch with CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Install remaining dependencies
   pip install -r requirements.txt
```

   **Option B: Create new environment from scratch:**
```powershell
   # Create environment from yml file
   conda env create -f environment.yml
   
   # Activate it
   conda activate segment-model
   
   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

   **Note:** Use `cu121` instead of `cu118` if you have CUDA 12.1+ drivers.

4. **Verify GPU access:**
```powershell
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Expected output: `CUDA available: True` and your GPU name.

## Quick Inference

Run inference on new MRI scans:

1. **Place NIfTI files in the inference folder:**
```powershell
# Copy your .nii.gz files to data/inference/
# They can be named anything (e.g., patient001.nii.gz, john-smith.nii.gz)
```

2. **Run batch inference on all files:**
```powershell
python scripts/run_inference_test_patient.py
```

That's it! The pipeline will:
- ✅ Auto-detect all `.nii.gz` files in `data/inference/`
- ✅ Auto-derive patient IDs from filenames
- ✅ Export 2D sagittal slices
- ✅ Run segmentation inference
- ✅ Save predictions in YOLO and JSON formats

For detailed usage and options, see [`docs/INFERENCE_USAGE.md`](docs/INFERENCE_USAGE.md).

---

## Development Workflow

*(Steps will be added as development progresses)*

---

## Contributing

- Use **Conventional Commits** (e.g., `feat:`, `fix:`, `docs:`)
- Follow semantic versioning
- Test changes on Windows before committing
- Update documentation for any new features

---

## License

*(Add your license here)*

---

## Contact

*(Add contact information)*