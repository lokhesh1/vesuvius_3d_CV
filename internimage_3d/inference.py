import os
import sys
import zipfile

import numpy as np
import torch
import torch.cuda.amp
import tifffile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.utils import get_positions
from common.dataset import open_zarr_array, discover_test_zarr
from internimage_3d.config import CFG
from internimage_3d.model import create_model


@torch.no_grad()
def predict_volume(model, vol_path, patch_size, stride, device):
    model.eval()
    vol = open_zarr_array(vol_path)
    D, H, W = vol.shape[:3]
    ps = patch_size
    pred_sum = np.zeros((D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    for z in get_positions(D, ps, stride):
        for y in get_positions(H, ps, stride):
            for x in get_positions(W, ps, stride):
                ze, ye, xe = min(z + ps, D), min(y + ps, H), min(x + ps, W)
                patch = np.array(vol[z:ze, y:ye, x:xe], dtype=np.float32) / 255.0
                pd_d, pd_h, pd_w = ps - patch.shape[0], ps - patch.shape[1], ps - patch.shape[2]
                if pd_d > 0 or pd_h > 0 or pd_w > 0:
                    patch = np.pad(patch, ((0, pd_d), (0, pd_h), (0, pd_w)))
                inp = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                with torch.cuda.amp.autocast(enabled=CFG.use_amp):
                    out = torch.sigmoid(model(inp)).cpu().numpy()[0, 0]
                pred_sum[z:ze, y:ye, x:xe] += out[:ze - z, :ye - y, :xe - x]
                count[z:ze, y:ye, x:xe] += 1

    return pred_sum / np.maximum(count, 1)


def generate_submission(model):
    if model is None:
        print("No trained model.")
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
        print(f"Predicting {name}...")
        pred = predict_volume(model, vp, CFG.patch_size, CFG.patch_size // 2, CFG.device)
        out_path = os.path.join(CFG.output_dir, "predictions", f"{name}.tif")
        tifffile.imwrite(out_path, (pred > 0.5).astype(np.uint8) * 255)
        pred_files.append(out_path)

    zip_path = os.path.join(CFG.output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in pred_files:
            zf.write(fp, os.path.basename(fp))
    print(f"Submission: {zip_path}")


if __name__ == "__main__":
    model = create_model().to(CFG.device)
    generate_submission(model)
