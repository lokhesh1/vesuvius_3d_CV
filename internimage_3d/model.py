import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG


class DCNv3Block3D(nn.Module):
    """DCNv3-inspired adaptive conv block for 3D data.

    Uses learnable spatial offsets with modulation for dynamic receptive fields.
    """

    def __init__(self, dim, groups=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(dim)

        self.dw_conv = nn.Conv3d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.offset_conv = nn.Sequential(
            nn.Conv3d(dim, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(dim, dim, 3, padding=1, groups=groups, bias=False),
        )
        self.modulation = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )
        self.value_conv = nn.Conv3d(dim, dim, 1)
        self.proj = nn.Conv3d(dim, dim, 1)

        self.norm2 = nn.InstanceNorm3d(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, dim, 1),
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        dw = self.dw_conv(x)
        offset_feat = self.offset_conv(dw)
        gate = self.modulation(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        value = self.value_conv(x)
        x = shortcut + self.proj(value * gate + offset_feat * (1 - gate))

        x = x + self.ffn(self.norm2(x))
        return x


class InternImageStage3D(nn.Module):
    def __init__(self, dim, depth, groups, downsample=True, in_dim=None):
        super().__init__()
        if in_dim is None:
            in_dim = dim
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_dim, dim, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm3d(dim),
            )
        elif in_dim != dim:
            self.downsample = nn.Conv3d(in_dim, dim, 1)

        self.blocks = nn.Sequential(*[DCNv3Block3D(dim, groups) for _ in range(depth)])

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return self.blocks(x)


class UPerNet3D(nn.Module):
    """UPerNet-style decoder for 3D feature maps with FPN."""

    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, out_channels, 1, bias=False),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True),
            )
            for ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True),
            )
            for _ in in_channels_list
        ])
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels * len(in_channels_list), out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="trilinear", align_corners=False
            )

        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        target_size = fpn_outs[0].shape[2:]
        upsampled = [F.interpolate(f, size=target_size, mode="trilinear", align_corners=False) for f in fpn_outs]
        return self.fusion(torch.cat(upsampled, dim=1))


class InternImage3D(nn.Module):
    """InternImage backbone adapted for 3D volumetric segmentation.

    Reference: Wang et al., "InternImage: Exploring Large-Scale Vision
    Foundation Models with Deformable Convolutions" (CVPR 2023).
    """

    def __init__(self, in_ch=1, out_ch=1, features=None, depths=None, groups=None):
        super().__init__()
        if features is None:
            features = [48, 96, 192, 384]
        if depths is None:
            depths = [2, 2, 4, 2]
        if groups is None:
            groups = [4, 8, 16, 32]

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, features[0], 7, stride=2, padding=3, bias=False),
            nn.InstanceNorm3d(features[0]),
            nn.LeakyReLU(inplace=True),
        )

        self.stages = nn.ModuleList()
        self.stages.append(InternImageStage3D(features[0], depths[0], groups[0], downsample=False))
        for i in range(1, len(features)):
            self.stages.append(InternImageStage3D(features[i], depths[i], groups[i],
                                                   downsample=True, in_dim=features[i - 1]))

        self.decoder = UPerNet3D(features, out_channels=128)
        self.head = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1, bias=False),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(inplace=True),
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
    return InternImage3D(in_ch=1, out_ch=1, features=CFG.features,
                          depths=CFG.depths, groups=CFG.groups)
