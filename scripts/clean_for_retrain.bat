@echo off
echo ======================================
echo Clean Project for Retraining
echo ======================================
echo.
echo This will remove:
echo - Old exported slices
echo - Old inference outputs
echo - Old QC overlays
echo - Old prediction masks
echo.
echo It will KEEP:
echo - Your training data (data\raw\)
echo - Previous training runs (outputs\runs\)
echo - Source code
echo.
pause

echo Removing old processed data...
rmdir /s /q data\derivatives\yolo_sagittal 2>nul
rmdir /s /q data\inference\slices 2>nul
rmdir /s /q outputs\inference 2>nul
rmdir /s /q outputs\qc 2>nul
rmdir /s /q outputs\masks 2>nul

echo.
echo ======================================
echo Cleanup complete!
echo ======================================
echo.
echo You can now run:
echo   python scripts\run_export_slices.py
echo.
pause