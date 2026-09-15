import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """Classic 3D UNet with encoder-decoder + skip connections.

    Reference: Cicek et al., "3D U-Net: Learning Dense Volumetric
    Segmentation from Sparse Annotation" (MICCAI 2016).
    """

    def __init__(self, in_ch=1, out_ch=1, features=None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        ch = in_ch
        for f in features:
            self.encoders.append(DoubleConv3D(ch, f))
            ch = f

        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv3D(f * 2, f))

        self.head = nn.Conv3d(features[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)


def create_model():
    return UNet3D(in_ch=1, out_ch=1, features=CFG.features)
