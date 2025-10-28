@echo off
REM Batch script to train YOLOv8 segmentation model on Windows

echo ======================================
echo YOLOv8 Spine Segmentation Training
echo ======================================
echo.

REM Activate conda environment
call conda activate segment-model
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not activate segment-model environment
    exit /b 1
)

REM Run training
python scripts\run_train.py

echo.
echo ======================================
echo Training script finished
echo ======================================
echo.
echo Check outputs/runs/ for results
pause