# SegFormer-B0 (2.5D) Baseline

## Overview

Baseline model for the **Vesuvius Challenge - Surface Detection** task using a SegFormer-B0 backbone adapted for volumetric data via 2.5D slice processing. Each z-slice is processed with K neighboring slices as context, and a lightweight 3D refinement head enforces inter-slice consistency.

## Architecture

**Model**: SegFormer-B0 (MiT-B0 encoder + MLP decoder) with 2.5D adaptation  
**Paper**: Xie et al., *"SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"* (NeurIPS 2021)  
**Framework**: Pure PyTorch + HuggingFace Transformers

### 2.5D Slice Processing
- Each z-slice receives context from **5 neighboring slices** (2 above, 2 below)
- Context slices projected from 5-channel to 3-channel (RGB-like) via 1x1 conv
- Processed in chunks of 32 slices for memory efficiency

### SegFormer-B0 Backbone
- 4-stage MiT (Mix Transformer) encoder
- Channel progression: 32 → 64 → 160 → 256
- Efficient self-attention with spatial reduction ratios: 8, 4, 2, 1
- MLP decoder with hidden dim 256

### 3D Refinement Head
- Two 3D conv layers (kernel: 5x3x3) with InstanceNorm + LeakyReLU
- Enforces inter-slice consistency in the z-direction
- Residual connection from 2D predictions

## Training Details

| Parameter | Value |
|-----------|-------|
| Patch size | 128³ |
| Stride | 64 |
| Batch size | 2 |
| Epochs | 25 |
| Optimizer | AdamW (lr=5e-4, wd=1e-4) |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Loss | BCE + Dice (50/50) |
| Precision | Mixed (FP16) |
| Context slices | 5 |
| Slice chunk size | 32 |

## Data Format

- **Input**: Zarr volumes (`.zarr`)
- **Labels**: Binary Zarr masks
- Training uses patch-based extraction with sliding window

## Inference

- Sliding-window inference with stride = patch_size / 2
- Overlapping predictions averaged for smooth output
- Binary threshold at 0.5

## Dependencies

- `transformers` (HuggingFace) for SegFormer implementation
- `zarr` for volumetric data I/O

## File Structure

```
segformer_25d/
├── config.py       # Hyperparameters (context_slices, slice_chunk, etc.)
├── model.py        # SegFormer2_5D (input_proj + SegFormer + refine3d)
├── train.py        # Training loop (pure PyTorch)
├── inference.py    # Sliding-window prediction + submission
└── README.md
```

## Usage

```bash
python -m segformer_25d.train
python -m segformer_25d.inference
```
