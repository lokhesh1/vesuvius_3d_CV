import os
import sys
import zipfile

import numpy as np
import torch
import torch.cuda.amp
import tifffile
from pathlib import Path
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.utils import get_positions
from common.dataset import open_zarr_array, discover_test_zarr
from depth_aware_3d.config import CFG
from depth_aware_3d.model import create_model


def augment_batch(patch, aug_idx):
    if aug_idx == 0:
        return patch
    elif aug_idx == 1:
        return torch.flip(patch, [2])
    elif aug_idx == 2:
        return torch.flip(patch, [3])
    elif aug_idx == 3:
        return torch.flip(patch, [4])
    elif aug_idx == 4:
        return torch.flip(patch, [2, 3])
    elif aug_idx == 5:
        return torch.flip(patch, [2, 4])
    elif aug_idx == 6:
        return torch.flip(patch, [3, 4])
    elif aug_idx == 7:
        return torch.flip(patch, [2, 3, 4])
    return patch


@torch.no_grad()
def predict_volume_tta(model, vol_path, patch_size, stride, device, use_tta=True):
    """Sliding-window inference with optional 3D TTA (flips + rotation ensembling)."""
    model.eval()
    vol = open_zarr_array(vol_path)
    D, H, W = vol.shape[:3]
    ps = patch_size
    pred_sum = np.zeros((D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    n_augs = 8 if use_tta else 1

    for z in get_positions(D, ps, stride):
        for y in get_positions(H, ps, stride):
            for x in get_positions(W, ps, stride):
                ze, ye, xe = min(z + ps, D), min(y + ps, H), min(x + ps, W)
                patch = np.array(vol[z:ze, y:ye, x:xe], dtype=np.float32) / 255.0
                pd_d, pd_h, pd_w = ps - patch.shape[0], ps - patch.shape[1], ps - patch.shape[2]
                if pd_d > 0 or pd_h > 0 or pd_w > 0:
                    patch = np.pad(patch, ((0, pd_d), (0, pd_h), (0, pd_w)))

                inp = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                preds = []
                for aug_i in range(n_augs):
                    aug_inp = augment_batch(inp, aug_i)
                    with torch.cuda.amp.autocast(enabled=CFG.use_amp):
                        out = torch.sigmoid(model(aug_inp))
                    out = augment_batch(out, aug_i)
                    preds.append(out.cpu().numpy()[0, 0])

                avg_pred = np.mean(preds, axis=0)
                avg_pred = avg_pred[:ze - z, :ye - y, :xe - x]
                pred_sum[z:ze, y:ye, x:xe] += avg_pred
                count[z:ze, y:ye, x:xe] += 1

    return pred_sum / np.maximum(count, 1)


def post_process(pred, min_size=500):
    """Remove small connected components from the prediction."""
    binary = (pred > 0.5).astype(np.uint8)
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return binary
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    for i, size in enumerate(sizes):
        if size < min_size:
            binary[labeled == (i + 1)] = 0
    return binary


def generate_submission(model):
    if model is None:
        return

    test_vols = discover_test_zarr(CFG.data_dir)
    if not test_vols:
        print("No test volumes found.")
        return

    model.to(CFG.device)
    os.makedirs(os.path.join(CFG.output_dir, "predictions"), exist_ok=True)
    pred_files = []

    for vp in test_vols:
        name = Path(vp).stem.replace(".zarr", "")
        print(f"Predicting {name} (TTA={CFG.use_tta})...")
        pred = predict_volume_tta(model, vp, CFG.patch_size, CFG.patch_size // 2,
                                   CFG.device, use_tta=CFG.use_tta)
        pred_clean = post_process(pred, min_size=500)
        out_path = os.path.join(CFG.output_dir, "predictions", f"{name}.tif")
        tifffile.imwrite(out_path, (pred_clean * 255).astype(np.uint8))
        pred_files.append(out_path)
        print(f"  Saved: {out_path} | shape: {pred_clean.shape}")

    zip_path = os.path.join(CFG.output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in pred_files:
            zf.write(fp, os.path.basename(fp))
    print(f"Submission: {zip_path}")


if __name__ == "__main__":
    model = create_model().to(CFG.device)
    generate_submission(model)
