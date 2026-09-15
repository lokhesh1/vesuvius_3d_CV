# InternImage-3D (Advanced)

## Overview

Advanced model for the **Vesuvius Challenge - Surface Detection** task using an InternImage backbone adapted for 3D volumetric segmentation. Features DCNv3-inspired adaptive convolutions with learnable spatial offsets and a UPerNet-style decoder.

## Architecture

**Model**: InternImage-3D  
**Paper**: Wang et al., *"InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions"* (CVPR 2023)  
**Framework**: PyTorch Lightning

### DCNv3-Inspired 3D Block
Each block replaces standard convolutions with adaptive convolutions:
1. **Depth-wise 3D conv**: Captures local structure efficiently
2. **Offset prediction network**: Learns dynamic spatial sampling offsets via grouped convolution
3. **Modulation gate**: Channel-wise attention for weighted aggregation between offset and value features
4. **Feed-forward network**: Channel mixing with GELU activation

### Backbone
- **Stem**: 7x7x7 strided conv (stride 2) for initial feature extraction
- **4 stages** with progressive downsampling (stride-2 conv between stages)
- Channel progression: 48 → 96 → 192 → 384
- Block depths: 2, 2, 4, 2
- Group counts: 4, 8, 16, 32

### UPerNet-style 3D Decoder
- Lateral connections (1x1 conv) from all 4 stages
- Top-down feature pyramid with trilinear upsampling
- FPN convolutions (3x3x3) for refinement
- Multi-scale fusion via concatenation + 3x3x3 conv

### Segmentation Head
- 128 → 64 channels (3x3x3 conv + InstanceNorm + LeakyReLU)
- 64 → 1 channel (1x1x1 conv)
- 2x trilinear upsampling to match input resolution

## Training Details

| Parameter | Value |
|-----------|-------|
| Patch size | 128³ |
| Stride | 64 |
| Batch size | 2 |
| Epochs | 30 |
| Optimizer | AdamW (lr=5e-4, wd=1e-4) |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Loss | BCE + Dice (50/50) |
| Precision | Mixed (FP16) |
| Early stopping | patience=7 on val_dice |
| Augmentations | 3D flips + 90° rotations |

## Data Format

- **Input**: Zarr volumes (`.zarr`)
- **Labels**: Binary Zarr masks

## Dependencies

- `pytorch-lightning`
- `zarr`

## File Structure

```
internimage_3d/
├── config.py       # Hyperparameters (features, depths, groups)
├── model.py        # DCNv3Block3D, InternImageStage3D, UPerNet3D, InternImage3D
├── train.py        # Lightning module + trainer setup
├── inference.py    # Sliding-window prediction + submission
└── README.md
```

## Usage

```bash
python -m internimage_3d.train
python -m internimage_3d.inference
```
