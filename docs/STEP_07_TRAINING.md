# Step 7: Training YOLOv8 Segmentation Model

## Overview

The `src/yolo/train.py` script trains YOLOv8-seg on the spine MRI dataset with optimized hyperparameters for medical imaging.

## Quick Start

### Run training
```powershell
python scripts\run_train.py
```

Or double-click: `scripts\train_yolo.bat`

## Model Sizes

| Model | Size | Speed | Accuracy | Recommended For |
|-------|------|-------|----------|-----------------|
| yolov8n-seg | Nano | Fastest | Lower | Quick experiments |
| yolov8s-seg | Small | Fast | Good | Limited data |
| yolov8m-seg | Medium | Balanced | Better | **Default choice** |
| yolov8l-seg | Large | Slow | Best | Large datasets |
| yolov8x-seg | Extra Large | Slowest | Best | Production |

**Default:** `yolov8m-seg.pt` (52MB, good balance)

## Training Parameters

### Default settings
```python
--data configs/yolo_sagittal.yaml
--model yolov8m-seg.pt
--epochs 100
--imgsz 640
--batch 8
--patience 50  # Early stopping
```

### Hyperparameters

**Optimizer:** AdamW
- Initial LR: 0.001
- Final LR: 0.00001
- Weight decay: 0.0005
- Warmup: 3 epochs

**Augmentation (medical imaging optimized):**
- Horizontal flip: 50%
- HSV augmentation: Moderate
- Translation/scale: Minimal
- **No rotation, shear, or perspective** (preserves anatomy)

**Early stopping:** Stops if no improvement for 50 epochs

## Custom Training

### Different model size
```powershell
python -m src.yolo.train --model yolov8s-seg.pt
```

### More epochs
```powershell
python -m src.yolo.train --epochs 200 --patience 100
```

### Larger images (if GPU memory allows)
```powershell
python -m src.yolo.train --imgsz 1280 --batch 4
```

### CPU training (slow)
```powershell
python -m src.yolo.train --device cpu
```

### Train from scratch (no pretrained weights)
```powershell
python -m src.yolo.train --no-pretrained
```

## Output Structure
```
outputs/runs/train_YYYYMMDD_HHMMSS/
├─ weights/
│  ├─ best.pt        # Best model (use this for inference)
│  └─ last.pt        # Last epoch
├─ results.png       # Training curves
├─ results.csv       # Metrics per epoch
├─ confusion_matrix.png
├─ val_batch0_labels.jpg
├─ val_batch0_pred.jpg
└─ args.yaml
```

## Monitoring Training

Watch the console output for:
- **Loss metrics:** Box, seg, cls, dfl (should decrease)
- **mAP metrics:** mAP50, mAP50-95 (should increase)
- **Validation predictions:** Check val_batch images

## Troubleshooting

### Out of memory error
```powershell
# Reduce batch size
python -m src.yolo.train --batch 4

# Or reduce image size
python -m src.yolo.train --imgsz 320
```

### Training too slow
- Check GPU is being used (should show CUDA device in output)
- Verify CUDA drivers are up to date
- Consider using smaller model (yolov8s-seg)

### Poor results with 29 images
- **Expected!** This is preliminary training with limited data
- Model will improve significantly when you add more annotated patients
- Focus on pipeline validation rather than model performance

## Next Steps

Step 8 will create the inference script to test the trained model.