# SegNeXt-3D (Advanced)

## Overview

Advanced model for the **Vesuvius Challenge - Surface Detection** task using a SegNeXt backbone adapted for 3D volumetric segmentation. Combines the MSCAN (Multi-Scale Convolutional Attention Network) encoder with a Hamburger decoder for efficient global context aggregation.

## Architecture

**Model**: SegNeXt-3D  
**Paper**: Guo et al., *"SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation"* (NeurIPS 2022)  
**Framework**: PyTorch Lightning

### MSCAN-3D Encoder (Multi-Scale Convolutional Attention Network)
Each attention block uses:
1. **Depth-wise 3D conv** (5x5x5): Local feature extraction
2. **Multi-scale strip convolutions**: Three parallel branches with axis-aligned large-kernel convolutions:
   - Depth strip: (7,1,1) → (11,1,1)
   - Height strip: (1,7,1) → (1,11,1)
   - Width strip: (1,1,7) → (1,1,11)
3. **1x1 projection**: Combines attention branches
4. **Depth-wise FFN**: Channel mixing with GELU activation

The strip convolutions enable large effective receptive fields without the computational cost of full 3D large-kernel convolutions.

### Backbone
- **Stem**: 7x7x7 strided conv (stride 2) with GELU
- **4 stages**: 32 → 64 → 160 → 256 channels
- Block depths: 3, 3, 5, 2
- Stride-2 downsampling between stages

### Hamburger Decoder (Matrix Decomposition)
Global context modeling via low-rank matrix decomposition:
1. Project all multi-scale features to a hidden dim (256)
2. Upsample and concatenate → squeeze to hidden dim
3. Reshape to 2D matrix and apply low-rank factorization (rank=64)
4. Reconstruct spatial features with global context
5. Residual connection + refinement conv

### Segmentation Head
- 256 → 64 channels (3x3x3 conv)
- 64 → 1 channel (1x1x1 conv)
- 2x trilinear upsampling

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

## Data Format

- **Input**: Zarr volumes (`.zarr`)
- **Labels**: Binary Zarr masks

## Dependencies

- `pytorch-lightning`
- `zarr`

## File Structure

```
segnext_3d/
├── config.py       # Hyperparameters (features, depths)
├── model.py        # MSCA3D, MSCANBlock3D, MSCANStage3D, HamburgerDecoder3D, SegNeXt3D
├── train.py        # Lightning module + trainer setup
├── inference.py    # Sliding-window prediction + submission
└── README.md
```

## Usage

```bash
python -m segnext_3d.train
python -m segnext_3d.inference
```
