# Papers & References — Vesuvius Challenge Surface Detection

## Competition

**Vesuvius Challenge — Surface Detection**
- Platform: [Kaggle](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)
- Task: 3D binary segmentation of papyrus surfaces in CT scan volumes (128³ zarr chunks, 8-bit intensity)
- Metric: Topology-aware linear blend rewarding voxel accuracy and surface connectivity (penalizes gaps, holes, sheet-switches, mergers)
- Prize pool: $100,000

---

## Notebook 1 — 3D UNet Baseline

**Paper**: Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016).
*3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.*
MICCAI 2016. [arXiv:1606.06650](https://arxiv.org/abs/1606.06650)

**Summary**: Extends the original 2D U-Net to 3D volumetric data. Uses an encoder-decoder architecture with 3D convolutions, max-pooling for downsampling, transposed convolutions for upsampling, and skip connections between corresponding encoder and decoder levels. Instance normalization and LeakyReLU activations are used for stable 3D training.

**Our implementation**: 4-level encoder (32→64→128→256 channels), 512-channel bottleneck, symmetric decoder with skip connections. Patch-based training on 128³ volumes with BCE+Dice loss.

---

## Notebook 2 — SegFormer 2.5D Baseline

**Paper**: Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021).
*SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.*
NeurIPS 2021. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)

**Summary**: Introduces a hierarchical Transformer encoder (Mix Transformer / MiT) without positional encodings, paired with a lightweight all-MLP decoder. MiT-B0 is the smallest variant with 4 stages of overlapping patch embeddings and efficient self-attention. Achieves strong performance with lower computational cost than prior transformer segmentation models.

**Our implementation**: 2.5D approach — processes each z-slice with K=5 neighboring slices as context channels. A 1×1 conv projects the multi-slice input to 3 channels for the pretrained SegFormer-B0. Predictions are stacked along z, then a lightweight 3D refinement head (two 3D conv layers) enforces inter-slice consistency.

---

## Notebook 3 — InternImage 3D (Advanced)

**Paper**: Wang, W., Dai, J., Chen, Z., Huang, Z., Li, Z., Zhu, X., Hu, X., Lu, T., Lu, L., Li, H., Wang, X., & Qiao, Y. (2023).
*InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions.*
CVPR 2023. [arXiv:2211.05778](https://arxiv.org/abs/2211.05778)

**Summary**: Proposes using Deformable Convolution v3 (DCNv3) as the core operator for a large-scale vision backbone. DCNv3 improves on prior deformable convolutions with: (1) group-wise multi-head design, (2) normalized modulation scalars via softmax, and (3) shared sampling offsets across groups. The resulting backbone achieves state-of-the-art across classification, detection, and segmentation, rivaling vision transformers while maintaining the inductive biases of convolutions.

**Our implementation**: Adapts InternImage for 3D volumetric segmentation. The DCNv3 block is approximated using depth-wise 3D convolutions with learnable offset modulation and channel-wise gating (exact 3D deformable convolutions require custom CUDA kernels). Architecture: stem + 4-stage backbone (48→96→192→384 channels, depths [2,2,4,2]) + UPerNet-style 3D decoder with feature pyramid fusion.

---

## Notebook 4 — SegNeXt 3D (Advanced)

**Paper**: Guo, M.-H., Lu, C.-Z., Hou, Q., Liu, Z., Cheng, M.-M., & Hu, S.-M. (2022).
*SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation.*
NeurIPS 2022. [arXiv:2209.08575](https://arxiv.org/abs/2209.08575)

**Summary**: Introduces the Multi-Scale Convolutional Attention Network (MSCAN) encoder that replaces self-attention with convolutional attention. Key innovation: multi-scale strip convolutions (1×K and K×1 decomposed large kernels) that capture long-range dependencies efficiently. Paired with a Hamburger decoder that uses matrix decomposition (NMF) for global context aggregation. Outperforms transformer-based methods while being more parameter-efficient.

**Our implementation**: Extends MSCAN to 3D with axis-aligned strip convolutions (1×1×K, 1×K×1, K×1×1 with K=7 and K=11) for efficient large-kernel attention across all three spatial axes. The Hamburger decoder uses low-rank matrix factorization (rank=64) for global 3D context. Architecture: stem + 4-stage MSCAN-3D (32→64→160→256, depths [3,3,5,2]) + Hamburger decoder.

---

## Notebook 5 — Proposed: Depth-Aware 3D Architecture

**Novel architecture** designed specifically for the surface detection task.

### Key innovations:

1. **Anisotropic 3D Convolutions**
   - Separate z-axis (5×1×1) and xy-plane (1×3×3) convolutions
   - Gated fusion learns the optimal blend per-channel
   - Motivation: CT scans often have different resolution/information density along depth vs. in-plane

2. **Cross-Slice Attention (CSA)**
   - Multi-head self-attention computed along the z-axis at each spatial position
   - Forces explicit modeling of inter-slice surface continuity
   - Related to: Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT, ICLR 2021) — applies attention mechanism along depth dimension

3. **Topology-Aware Loss**
   - Combines BCE + Dice + differentiable connectivity penalty
   - Connectivity penalty uses morphological operations (erosion → dilation) to detect and penalize disconnected fragments
   - Motivation: competition metric explicitly rewards topological correctness
   - Related to: Hu et al., *Topology-Preserving Deep Image Segmentation* (NeurIPS 2019) [arXiv:1906.05404](https://arxiv.org/abs/1906.05404)

4. **Test-Time Augmentation (TTA)**
   - 8-fold augmentation: all combinations of 3D axis flips
   - Predictions averaged for robust inference
   - Post-processing: connected-component filtering removes small isolated fragments (<500 voxels)

### Architecture:
- Stem: two 3D convolutions (stride-2 + stride-1)
- 4-stage encoder with DepthAware blocks (32→64→128→256, depths [2,2,3,2])
- Multi-scale feature pyramid decoder with progressive upsampling
- CosineAnnealingWarmRestarts scheduler for extended training

---

## Additional References

- **nnU-Net**: Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021).
  *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.*
  Nature Methods, 18, 203–211. [arXiv:1904.08128](https://arxiv.org/abs/1904.08128)
  - Used by winning solutions (1st, 2nd, 7th place) in ensemble configurations

- **Vesuvius Challenge**: [scrollprize.org](https://scrollprize.org/)
  - Open data repository: `s3://vesuvius-challenge-open-data/`
  - Data formats: OME-Zarr (primary), TIFF stacks, 128³ voxel chunks, 8-bit intensity
  - Python package: [vesuvius (PyPI)](https://pypi.org/project/vesuvius/)

- **Competition solutions**:
  - 1st place: nnU-Net ensemble (128/192/256 patch) + post-processing
  - 7th place: Two nnU-Net models + TTA + post-processing
  - Common theme: topology-aware post-processing is critical for score
