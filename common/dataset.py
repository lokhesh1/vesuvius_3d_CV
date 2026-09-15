import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from .utils import get_positions


# ---------------------------------------------------------------------------
# Volume I/O helpers
# ---------------------------------------------------------------------------

def open_volume_tif(path):
    import tifffile
    return tifffile.imread(str(path))


def open_zarr_array(path):
    import zarr
    z = zarr.open(str(path), mode="r")
    if hasattr(z, "shape"):
        return z
    if "0" in z:
        return z["0"]
    keys = sorted(z.keys())
    if not keys:
        raise ValueError(f"No arrays found in Zarr store: {path}")
    return z[keys[0]]
    return z


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def discover_data_tif(data_dir, split="train"):
    base = Path(data_dir)
    img_dir = base / f"{split}_images"
    lbl_dir = base / f"{split}_labels"

    pairs = []
    if img_dir.exists() and lbl_dir.exists():
        for img_path in sorted(img_dir.glob("*.tif")):
            lbl_path = lbl_dir / img_path.name
            if lbl_path.exists():
                pairs.append((str(img_path), str(lbl_path)))

    print(f"[{split}] Found {len(pairs)} image-label pairs")
    print(f"  images dir: {img_dir}")
    print(f"  labels dir: {lbl_dir}")
    if pairs:
        print(f"  example: {Path(pairs[0][0]).name}")
    return pairs


def discover_test_tif(data_dir):
    base = Path(data_dir)
    test_dir = base / "test_images"
    if not test_dir.exists():
        print(f"[test] test_images dir not found at {test_dir}")
        return []
    images = [str(f) for f in sorted(test_dir.glob("*.tif"))]
    print(f"[test] Found {len(images)} test images in {test_dir}")
    return images


def discover_data_zarr(data_dir, split="train"):
    base = Path(data_dir)
    image_dir = base / f"{split}_images"
    label_dir = base / f"{split}_labels"
    if image_dir.exists() and label_dir.exists():
        pairs = []
        for image_path in sorted(image_dir.glob("*.zarr")):
            label_path = label_dir / image_path.name
            if label_path.exists():
                pairs.append((str(image_path), str(label_path)))
        print(f"[{split}] Found {len(pairs)} image-label pairs")
        print(f"  images dir: {image_dir}")
        print(f"  labels dir: {label_dir}")
        if pairs:
            print(f"  example: {Path(pairs[0][0]).name}")
        return pairs

    split_dir = base / split
    if not split_dir.exists():
        split_dir = base
    pairs = []

    for d in sorted(split_dir.iterdir()) if split_dir.is_dir() else []:
        if not d.is_dir():
            continue
        vols = list(d.glob("volume*.zarr")) + list(d.glob("scan*.zarr"))
        lbls = list(d.glob("mask*.zarr")) + list(d.glob("label*.zarr"))
        if vols and lbls:
            pairs.append((str(vols[0]), str(lbls[0])))

    if not pairs:
        zarr_files = sorted(split_dir.glob("*.zarr"))
        vol_files = [f for f in zarr_files if "label" not in f.stem and "mask" not in f.stem]
        for vf in vol_files:
            for suffix in ["_mask", "_label", "_seg"]:
                lf = vf.parent / (vf.stem + suffix + ".zarr")
                if lf.exists():
                    pairs.append((str(vf), str(lf)))
                    break

    if not pairs:
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if img_dir.exists() and lbl_dir.exists():
            for vf in sorted(img_dir.glob("*.zarr")):
                lf = lbl_dir / vf.name
                if not lf.exists():
                    lf = lbl_dir / vf.name.replace("volume", "label")
                if lf.exists():
                    pairs.append((str(vf), str(lf)))

    print(f"[{split}] Found {len(pairs)} volume-label pairs in {split_dir}")
    for v, l in pairs:
        print(f"  vol: {v}")
        print(f"  lbl: {l}")
    return pairs


def discover_test_zarr(data_dir):
    base = Path(data_dir)
    image_dir = base / "test_images"
    if image_dir.exists():
        volumes = [str(f) for f in sorted(image_dir.glob("*.zarr"))]
        print(f"[test] Found {len(volumes)} Zarr volumes in {image_dir}")
        return volumes

    test_dir = base / "test"
    if not test_dir.exists():
        test_dir = base
    volumes = []
    for d in sorted(test_dir.iterdir()) if test_dir.is_dir() else []:
        if d.is_dir():
            vols = list(d.glob("volume*.zarr")) + list(d.glob("scan*.zarr")) + list(d.glob("*.zarr"))
            if vols:
                volumes.append(str(vols[0]))
    if not volumes:
        volumes = [str(f) for f in sorted(test_dir.glob("*.zarr"))
                    if "label" not in f.stem and "mask" not in f.stem]
    print(f"[test] Found {len(volumes)} test volumes")
    return volumes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VesuviusDataset3D(Dataset):
    """Patch-based 3D dataset for volumetric segmentation.

    Args:
        vol_paths: list of volume file paths
        lbl_paths: list of label file paths
        patch_size: cube side length for patches
        stride: stride between patches
        augment: enable random flips and rotations
        backend: "tif" or "zarr"
    """

    def __init__(self, vol_paths, lbl_paths, patch_size=128, stride=64,
                 augment=False, backend="zarr"):
        self.patch_size = patch_size
        self.augment = augment
        self.volumes = []
        self.labels = []
        self.patches = []

        opener = open_volume_tif if backend == "tif" else open_zarr_array

        for vi, (vp, lp) in enumerate(zip(vol_paths, lbl_paths)):
            vol = opener(vp)
            lbl = opener(lp)
            self.volumes.append(vol)
            self.labels.append(lbl)

            D, H, W = vol.shape[:3]
            ps = patch_size
            for z in get_positions(D, ps, stride):
                for y in get_positions(H, ps, stride):
                    for x in get_positions(W, ps, stride):
                        self.patches.append((vi, z, y, x))

        print(f"Dataset: {len(self.patches)} patches from {len(self.volumes)} volumes")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        vi, z, y, x = self.patches[idx]
        ps = self.patch_size
        vol = self.volumes[vi]
        lbl = self.labels[vi]
        D, H, W = vol.shape[:3]

        ze, ye, xe = min(z + ps, D), min(y + ps, H), min(x + ps, W)
        v = np.array(vol[z:ze, y:ye, x:xe], dtype=np.float32)
        l = np.array(lbl[z:ze, y:ye, x:xe], dtype=np.float32)

        pd = ps - v.shape[0]
        ph = ps - v.shape[1]
        pw = ps - v.shape[2]
        if pd > 0 or ph > 0 or pw > 0:
            v = np.pad(v, ((0, pd), (0, ph), (0, pw)), mode="constant")
            l = np.pad(l, ((0, pd), (0, ph), (0, pw)), mode="constant")

        v = v / 255.0
        l = (l > 0.5).astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                v, l = v[::-1].copy(), l[::-1].copy()
            if random.random() > 0.5:
                v, l = v[:, ::-1].copy(), l[:, ::-1].copy()
            if random.random() > 0.5:
                v, l = v[:, :, ::-1].copy(), l[:, :, ::-1].copy()
            k = random.randint(0, 3)
            if k:
                v = np.rot90(v, k, axes=(1, 2)).copy()
                l = np.rot90(l, k, axes=(1, 2)).copy()

        v = torch.from_numpy(v).unsqueeze(0)
        l = torch.from_numpy(l).unsqueeze(0)
        return v, l
