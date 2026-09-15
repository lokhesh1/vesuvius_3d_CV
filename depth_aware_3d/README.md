# Proposed: Depth-Aware 3D Surface Detection

## Overview

The proposed model for the **Vesuvius Challenge - Surface Detection** task. A custom 3D architecture with depth-aware processing and topology-guided training, designed specifically for detecting surfaces in CT scan volumes.

## Architecture

**Model**: DepthAwareNet3D (custom)  
**Framework**: PyTorch Lightning

### Novel Contributions

#### 1. Anisotropic 3D Convolutions
Separate kernels for depth (z) and spatial (xy) dimensions, reflecting the different information density along CT scan axes:
- **Z-conv**: (5,1,1) kernel captures inter-slice relationships
- **XY-conv**: (1,3,3) kernel captures in-plane features
- **Gated fusion**: Learned channel-wise gate combines both pathways based on global average pooling

#### 2. Cross-Slice Attention (CSA)
Multi-head self-attention computed along the z-axis at each spatial position:
- Forces explicit modeling of inter-slice relationships
- Promotes surface continuity along depth
- 4 attention heads, scaled dot-product attention
- Residual connection for stable training

#### 3. Multi-Scale 3D Feature Pyramid Decoder
Aggregates features at 4 scales for both fine detail and global context:
- Lateral connections from all encoder stages
- Top-down pathway with trilinear upsampling
- Smoothing convolutions per level
- Concatenation + double 3x3x3 conv for fusion

#### 4. Topology-Aware Loss
Combines three loss terms to produce topology-clean predictions:
- **BCE loss** (50%): Standard pixel-wise classification
- **Dice loss** (40%): Overlap-based metric for class imbalance
- **Connectivity penalty** (10%): Differentiable morphological opening that penalizes fragmented, disconnected predictions

#### 5. Test-Time Augmentation (TTA)
8-fold augmentation ensembling at inference:
- All 3D axis flip combinations (2³ = 8 augmentations)
- Predictions averaged for robust output

### Backbone
- **Stem**: Double 3D conv (7x7x7 + 3x3x3) with stride 2
- **4 stages**: 32 → 64 → 128 → 256 channels
- Block depths: 2, 2, 3, 2
- Each block: AnisotropicConv3D + CrossSliceAttention + FFN

### Post-Processing
- Connected component analysis (scipy.ndimage)
- Removes small disconnected regions (< 500 voxels)

## Training Details

| Parameter | Value |
|-----------|-------|
| Patch size | 128³ |
| Stride | 64 |
| Batch size | 2 |
| Epochs | 40 |
| Optimizer | AdamW (lr=3e-4, wd=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| Loss | BCE (50%) + Dice (40%) + Topology (10%) |
| Precision | Mixed (FP16) |
| Early stopping | patience=10 on val_dice |
| CSA heads | 4 |
| TTA | 8-fold (all 3D flips) |

## Data Format

- **Input**: Zarr volumes (`.zarr`)
- **Labels**: Binary Zarr masks

## Dependencies

- `pytorch-lightning`
- `zarr`
- `scipy` (for post-processing)

## File Structure

```
depth_aware_3d/
├── config.py       # Hyperparameters (csa_heads, topo_weight, use_tta)
├── model.py        # AnisotropicConv3D, CrossSliceAttention, DepthAwareNet3D
├── loss.py         # TopologyAwareLoss (BCE + Dice + connectivity penalty)
├── train.py        # Lightning module + trainer setup
├── inference.py    # TTA inference + post-processing + submission
└── README.md
```

## Usage

```bash
python -m depth_aware_3d.train
python -m depth_aware_3d.inference
```
