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
ai-spine-seg/
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
   git clone <your-repo-url> ai-spine-seg
   cd ai-spine-seg
```

2. **Activate your existing conda environment:**
```powershell
   conda activate spineai
```

3. **Install dependencies** (see Step 2 setup instructions - coming next)

4. **Verify GPU access:**
```powershell
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

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