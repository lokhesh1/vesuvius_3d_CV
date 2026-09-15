import os
import sys
import random

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.utils import seed_everything
from common.dataset import VesuviusDataset3D, discover_data_tif, discover_data_zarr
from common.loss import DiceBCELoss, compute_dice
from unet3d.config import CFG
from unet3d.model import create_model


def amp_is_enabled(device):
    return CFG.use_amp and str(device).startswith("cuda") and torch.cuda.is_available()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_dice, n = 0, 0, 0
    amp_enabled = amp_is_enabled(device)
    for batch_idx, (volume, label) in enumerate(loader):
        volume, label = volume.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=amp_enabled):
            pred = model(volume)
            loss = criterion(pred, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_dice += compute_dice(pred.detach(), label)
        n += 1
        if batch_idx % 20 == 0:
            print(f"  batch {batch_idx}/{len(loader)} | loss: {loss.item():.4f}")
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, n = 0, 0, 0
    amp_enabled = amp_is_enabled(device)
    for volume, label in loader:
        volume, label = volume.to(device), label.to(device)
        with autocast(device_type="cuda", enabled=amp_enabled):
            pred = model(volume)
            loss = criterion(pred, label)
        total_loss += loss.item()
        total_dice += compute_dice(pred, label)
        n += 1
    return total_loss / n, total_dice / n


def main_train():
    seed_everything(CFG.seed)
    discover = discover_data_zarr if CFG.data_backend == "zarr" else discover_data_tif
    pairs = discover(CFG.data_dir, "train")
    if not pairs:
        print("No training data found. Check CFG.data_dir path.")
        return None

    vol_paths = [p[0] for p in pairs]
    lbl_paths = [p[1] for p in pairs]

    n = len(vol_paths)
    n_val = max(1, int(n * CFG.val_ratio))
    indices = list(range(n))
    random.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    if n == 1:
        train_idx = [0]
        val_idx = [0]

    train_vols = [vol_paths[i] for i in train_idx]
    train_lbls = [lbl_paths[i] for i in train_idx]
    val_vols = [vol_paths[i] for i in val_idx]
    val_lbls = [lbl_paths[i] for i in val_idx]

    train_ds = VesuviusDataset3D(
        train_vols, train_lbls, CFG.patch_size, CFG.stride,
        augment=True, backend=CFG.data_backend
    )
    val_ds = VesuviusDataset3D(
        val_vols, val_lbls, CFG.patch_size, CFG.patch_size,
        augment=False, backend=CFG.data_backend
    )

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True)

    model = create_model().to(CFG.device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.num_epochs, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=amp_is_enabled(CFG.device))

    best_dice = 0
    for epoch in range(CFG.num_epochs):
        print(f"\nEpoch {epoch+1}/{CFG.num_epochs}")
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, scaler, CFG.device)
        val_loss, val_dice = validate(model, val_loader, criterion, CFG.device)
        scheduler.step()

        print(f"  train loss: {train_loss:.4f} | dice: {train_dice:.4f}")
        print(f"  val   loss: {val_loss:.4f} | dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(CFG.output_dir, "best_model.pth"))
            print(f"  -> Saved best model (dice={best_dice:.4f})")

    print(f"\nTraining complete. Best val dice: {best_dice:.4f}")
    return model


if __name__ == "__main__":
    main_train()
