import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from .config import CFG


class SegFormer2_5D(nn.Module):
    """SegFormer-B0 adapted for volumetric data via 2.5D slice processing.

    Reference: Xie et al., "SegFormer: Simple and Efficient Design for
    Semantic Segmentation with Transformers" (NeurIPS 2021).

    Each z-slice is processed with K neighboring slices as multi-channel input.
    A lightweight 3D refinement head enforces inter-slice consistency.
    """

    def __init__(self, context_slices=5, slice_chunk=32):
        super().__init__()
        self.ctx = context_slices
        self.slice_chunk = slice_chunk

        self.input_proj = nn.Sequential(
            nn.Conv2d(context_slices, 3, 1, bias=False),
            nn.BatchNorm2d(3),
        )

        config = SegformerConfig(
            num_channels=3,
            num_labels=1,
            num_encoder_blocks=4,
            depths=[2, 2, 2, 2],
            hidden_sizes=[32, 64, 160, 256],
            decoder_hidden_size=256,
            sr_ratios=[8, 4, 2, 1],
            num_attention_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            mlp_ratios=[4, 4, 4, 4],
        )
        self.segformer = SegformerForSemanticSegmentation(config)

        self.refine3d = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 1, 1),
        )

    def process_slices(self, volume_3d):
        B, C, D, H, W = volume_3d.shape
        half = self.ctx // 2
        outputs = []

        for start in range(0, D, self.slice_chunk):
            end = min(start + self.slice_chunk, D)
            batch_slices = []

            for z in range(start, end):
                z0 = max(0, z - half)
                z1 = z0 + self.ctx
                if z1 > D:
                    z1 = D
                    z0 = max(0, z1 - self.ctx)
                slices = volume_3d[:, 0, z0:z1, :, :]
                if slices.shape[1] < self.ctx:
                    pad = self.ctx - slices.shape[1]
                    slices = F.pad(slices, (0, 0, 0, 0, 0, pad))
                batch_slices.append(slices)

            batch = torch.cat(batch_slices, dim=0)
            proj = self.input_proj(batch)
            logits = self.segformer(pixel_values=proj).logits
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            logits = logits.view(B, end - start, 1, H, W)
            outputs.append(logits)

        return torch.cat(outputs, dim=1).permute(0, 2, 1, 3, 4)

    def forward(self, x):
        slice_pred = self.process_slices(x)
        refined = self.refine3d(slice_pred)
        return refined + slice_pred


def create_model():
    return SegFormer2_5D(context_slices=CFG.context_slices, slice_chunk=CFG.slice_chunk)
