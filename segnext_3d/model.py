import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG


class MSCA3D(nn.Module):
    """Multi-Scale Convolutional Attention for 3D.

    Uses axis-aligned strip convolutions (1x1xK, 1xKx1, Kx1x1)
    for efficient large-kernel attention.
    """

    def __init__(self, dim):
        super().__init__()
        self.dw_conv = nn.Conv3d(dim, dim, 5, padding=2, groups=dim, bias=False)

        self.strip_d = nn.Sequential(
            nn.Conv3d(dim, dim, (7, 1, 1), padding=(3, 0, 0), groups=dim, bias=False),
            nn.Conv3d(dim, dim, (11, 1, 1), padding=(5, 0, 0), groups=dim, bias=False),
        )
        self.strip_h = nn.Sequential(
            nn.Conv3d(dim, dim, (1, 7, 1), padding=(0, 3, 0), groups=dim, bias=False),
            nn.Conv3d(dim, dim, (1, 11, 1), padding=(0, 5, 0), groups=dim, bias=False),
        )
        self.strip_w = nn.Sequential(
            nn.Conv3d(dim, dim, (1, 1, 7), padding=(0, 0, 3), groups=dim, bias=False),
            nn.Conv3d(dim, dim, (1, 1, 11), padding=(0, 0, 5), groups=dim, bias=False),
        )

        self.proj = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        attn = self.dw_conv(x)
        attn = self.strip_d(attn) + self.strip_h(attn) + self.strip_w(attn)
        return x * self.proj(attn)


class MSCANBlock3D(nn.Module):
    """MSCAN block with attention and FFN."""

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(dim)
        self.attn = MSCA3D(dim)
        self.norm2 = nn.InstanceNorm3d(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, hidden, 1),
            nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv3d(hidden, dim, 1),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MSCANStage3D(nn.Module):
    def __init__(self, in_dim, out_dim, depth, downsample=True):
        super().__init__()
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm3d(out_dim),
            )
        elif in_dim != out_dim:
            self.downsample = nn.Conv3d(in_dim, out_dim, 1)
        self.blocks = nn.Sequential(*[MSCANBlock3D(out_dim) for _ in range(depth)])

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return self.blocks(x)


class HamburgerDecoder3D(nn.Module):
    """Hamburger decoder using low-rank matrix decomposition for global context.

    Reshapes features to 2D matrix, applies low-rank approximation to
    capture global patterns, then reconstructs and fuses with skip features.
    """

    def __init__(self, in_channels_list, hidden_dim=256, rank=64):
        super().__init__()
        self.input_projs = nn.ModuleList([
            nn.Sequential(nn.Conv3d(ch, hidden_dim, 1, bias=False), nn.InstanceNorm3d(hidden_dim))
            for ch in in_channels_list
        ])
        self.rank = rank

        self.squeeze = nn.Conv3d(hidden_dim * len(in_channels_list), hidden_dim, 1, bias=False)

        self.mat_U = nn.Linear(hidden_dim, rank, bias=False)
        self.mat_V = nn.Linear(rank, hidden_dim, bias=False)

        self.norm = nn.InstanceNorm3d(hidden_dim)
        self.out_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_dim),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, features):
        target_size = features[0].shape[2:]
        projected = []
        for proj, feat in zip(self.input_projs, features):
            p = proj(feat)
            p = F.interpolate(p, size=target_size, mode="trilinear", align_corners=False)
            projected.append(p)

        fused = self.squeeze(torch.cat(projected, dim=1))

        B, C, D, H, W = fused.shape
        x_flat = fused.reshape(B, C, -1).permute(0, 2, 1)
        low_rank = self.mat_V(F.relu(self.mat_U(x_flat)))
        global_ctx = low_rank.permute(0, 2, 1).reshape(B, C, D, H, W)

        return self.out_conv(self.norm(fused + global_ctx))


class SegNeXt3D(nn.Module):
    """SegNeXt backbone (MSCAN encoder + Hamburger decoder) adapted for 3D.

    Reference: Guo et al., "SegNeXt: Rethinking Convolutional Attention
    Design for Semantic Segmentation" (NeurIPS 2022).
    """

    def __init__(self, in_ch=1, out_ch=1, features=None, depths=None):
        super().__init__()
        if features is None:
            features = [32, 64, 160, 256]
        if depths is None:
            depths = [3, 3, 5, 2]

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, features[0], 7, stride=2, padding=3, bias=False),
            nn.InstanceNorm3d(features[0]),
            nn.GELU(),
        )

        self.stages = nn.ModuleList()
        self.stages.append(MSCANStage3D(features[0], features[0], depths[0], downsample=False))
        for i in range(1, len(features)):
            self.stages.append(MSCANStage3D(features[i - 1], features[i], depths[i], downsample=True))

        self.decoder = HamburgerDecoder3D(features, hidden_dim=256, rank=64)

        self.head = nn.Sequential(
            nn.Conv3d(256, 64, 3, padding=1, bias=False),
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
    return SegNeXt3D(in_ch=1, out_ch=1, features=CFG.features, depths=CFG.depths)
