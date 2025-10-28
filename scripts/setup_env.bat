@echo off
REM Setup script for segment-model conda environment on Windows

echo ======================================
echo Spine Segmentation Environment Setup
echo ======================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda not found. Please install Miniconda or Anaconda.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

echo [1/4] Activating segment-model environment...
call conda activate segment-model
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not activate segment-model environment.
    echo Run: conda env create -f environment.yml
    exit /b 1
)

echo [2/4] Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo [3/4] Installing remaining dependencies...
pip install -r requirements.txt

echo [4/4] Verifying installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

echo.
echo ======================================
echo Setup complete!
echo ======================================
echo.
echo To activate the environment:
echo   conda activate segment-model
echo.