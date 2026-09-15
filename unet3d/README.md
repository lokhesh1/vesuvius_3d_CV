# 3D UNet Baseline

## Overview

Baseline model for the **Vesuvius Challenge - Surface Detection** task using a classic 3D UNet architecture with encoder-decoder structure and skip connections.

## Architecture

**Model**: 3D UNet  
**Paper**: Cicek et al., *"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"* (MICCAI 2016)  
**Framework**: Pure PyTorch

### Encoder
- 4 encoder stages: 32 → 64 → 128 → 256 channels
- Each stage: two 3D convolutions with InstanceNorm3d + LeakyReLU
- MaxPool3d (stride 2) between stages

### Bottleneck
- 512-channel DoubleConv3D block

### Decoder
- 4 decoder stages with transposed convolutions (stride 2) for upsampling
- Skip connections concatenated from corresponding encoder stages
- DoubleConv3D at each level to refine features

### Output
- 1x1x1 convolution producing single-channel logits

## Training Details

| Parameter | Value |
|-----------|-------|
| Patch size | 128³ |
| Stride | 64 |
| Batch size | 2 |
| Epochs | 30 |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Loss | BCE + Dice (50/50) |
| Precision | Mixed (FP16) |
| Augmentations | 3D flips (all axes) + 90° rotations (xy-plane) |

## Data Format

- **Input**: Multi-page TIFF volumes (`.tif`)
- **Labels**: Binary TIFF masks (surface = 1, background = 0)
- Training uses patch-based extraction with sliding window

## Inference

- Sliding-window inference with stride = patch_size / 2
- Overlapping predictions averaged for smooth output
- Binary threshold at 0.5
- Output zipped as `submission.zip`

## File Structure

```
unet3d/
├── config.py       # Hyperparameters and paths
├── model.py        # UNet3D architecture (DoubleConv3D, UNet3D)
├── train.py        # Training loop (pure PyTorch)
├── inference.py    # Sliding-window prediction + submission
└── README.md
```

## Usage

```bash
python -m unet3d.train
python -m unet3d.inference
```
