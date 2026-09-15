import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG


class AnisotropicConv3D(nn.Module):
    """Separate depth (z) and spatial (xy) convolutions with gated fusion.

    Reflects the different information density along CT scan axes by
    using independent receptive fields for z vs. xy dimensions.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.z_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (5, 1, 1), padding=(2, 0, 0), bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )
        self.xy_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(out_ch * 2, out_ch),
            nn.Sigmoid(),
        )
        self.combine = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        z_feat = self.z_conv(x)
        xy_feat = self.xy_conv(x)
        gate_input = torch.cat([
            z_feat.mean(dim=(2, 3, 4)),
            xy_feat.mean(dim=(2, 3, 4)),
        ], dim=1)
        g = self.gate(gate_input).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fused = g * z_feat + (1 - g) * xy_feat
        return self.combine(fused)


class CrossSliceAttention(nn.Module):
    """Multi-head self-attention along the depth (z) axis.

    Forces the network to model explicit relationships between slices,
    promoting surface continuity along depth.
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.InstanceNorm3d(dim)
        self.qkv = nn.Conv3d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        shortcut = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, D, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = q.permute(0, 1, 4, 3, 2, 5, 6).reshape(B * self.num_heads * H * W, D, self.head_dim)
        k = k.permute(0, 1, 4, 3, 2, 5, 6).reshape(B * self.num_heads * H * W, D, self.head_dim)
        v = v.permute(0, 1, 4, 3, 2, 5, 6).reshape(B * self.num_heads * H * W, D, self.head_dim)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)

        out = out.reshape(B, self.num_heads, H, W, D, self.head_dim)
        out = out.permute(0, 1, 5, 4, 2, 3).reshape(B, C, D, H, W)
        return shortcut + self.proj(out)


class DepthAwareBlock(nn.Module):
    """Combined anisotropic conv + cross-slice attention + FFN."""

    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.aniso_conv = AnisotropicConv3D(dim, dim)
        self.csa = CrossSliceAttention(dim, num_heads)
        self.norm = nn.InstanceNorm3d(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, dim, 1),
        )

    def forward(self, x):
        x = x + self.aniso_conv(x)
        x = self.csa(x)
        x = x + self.ffn(self.norm(x))
        return x


class DepthAwareStage(nn.Module):
    def __init__(self, in_dim, out_dim, depth, num_heads, downsample=True):
        super().__init__()
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm3d(out_dim),
            )
        elif in_dim != out_dim:
            self.downsample = nn.Conv3d(in_dim, out_dim, 1)
        self.blocks = nn.Sequential(*[DepthAwareBlock(out_dim, num_heads) for _ in range(depth)])

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return self.blocks(x)


class MultiScaleDecoder3D(nn.Module):
    """Multi-scale feature pyramid decoder with progressive upsampling."""

    def __init__(self, in_channels_list, out_ch=128):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Sequential(nn.Conv3d(c, out_ch, 1, bias=False), nn.InstanceNorm3d(out_ch), nn.GELU())
            for c in in_channels_list
        ])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
                          nn.InstanceNorm3d(out_ch), nn.GELU())
            for _ in in_channels_list
        ])
        self.fuse = nn.Sequential(
            nn.Conv3d(out_ch * len(in_channels_list), out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, features):
        lats = [l(f) for l, f in zip(self.lateral, features)]
        for i in range(len(lats) - 1, 0, -1):
            lats[i - 1] = lats[i - 1] + F.interpolate(lats[i], size=lats[i - 1].shape[2:],
                                                         mode="trilinear", align_corners=False)
        smoothed = [s(l) for s, l in zip(self.smooth, lats)]
        target = smoothed[0].shape[2:]
        up = [F.interpolate(s, size=target, mode="trilinear", align_corners=False) for s in smoothed]
        return self.fuse(torch.cat(up, dim=1))


class DepthAwareNet3D(nn.Module):
    """Proposed depth-aware 3D architecture for surface detection.

    Novel contributions:
    - Anisotropic 3D convolutions with separate z / xy receptive fields
    - Cross-Slice Attention (CSA) for inter-slice consistency
    - Multi-Scale 3D Feature Pyramid decoder
    - Designed for topology-aware training (see loss.py)
    """

    def __init__(self, in_ch=1, out_ch=1, features=None, depths=None, num_heads=4):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        if depths is None:
            depths = [2, 2, 3, 2]

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, features[0], 7, stride=2, padding=3, bias=False),
            nn.InstanceNorm3d(features[0]),
            nn.GELU(),
            nn.Conv3d(features[0], features[0], 3, padding=1, bias=False),
            nn.InstanceNorm3d(features[0]),
            nn.GELU(),
        )

        self.stages = nn.ModuleList()
        self.stages.append(DepthAwareStage(features[0], features[0], depths[0], num_heads, downsample=False))
        for i in range(1, len(features)):
            self.stages.append(DepthAwareStage(features[i - 1], features[i], depths[i], num_heads, downsample=True))

        self.decoder = MultiScaleDecoder3D(features, out_ch=128)

        self.head = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1, bias=False),
            nn.InstanceNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, out_ch, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        decoded = self.decoder(features)
        out = F.interpolate(decoded, scale_factor=2, mode="trilinear", align_corners=False)
        return self.head(out)


def create_model():
    return DepthAwareNet3D(in_ch=1, out_ch=1, features=CFG.features,
                            depths=[2, 2, 3, 2], num_heads=CFG.csa_heads)
