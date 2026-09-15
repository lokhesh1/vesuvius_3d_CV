import os
import sys
import random

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.utils import seed_everything
from common.dataset import VesuviusDataset3D, discover_data_zarr
from common.loss import DiceBCELoss, compute_dice
from segnext_3d.config import CFG
from segnext_3d.model import create_model


class SurfaceDetectionModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_model()
        self.criterion = DiceBCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        volume, label = batch
        pred = self.model(volume)
        loss = self.criterion(pred, label)
        dice = compute_dice(pred.detach(), label)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dice", dice, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        volume, label = batch
        pred = self.model(volume)
        loss = self.criterion(pred, label)
        dice = compute_dice(pred.detach(), label)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_dice", dice, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.num_epochs, eta_min=1e-6)
        return [optimizer], [scheduler]


def main_train():
    seed_everything(CFG.seed)
    pairs = discover_data_zarr(CFG.data_dir, "train")
    if not pairs:
        print("No training data found.")
        return None

    vol_paths = [p[0] for p in pairs]
    lbl_paths = [p[1] for p in pairs]

    n = len(vol_paths)
    n_val = max(1, int(n * CFG.val_ratio))
    indices = list(range(n))
    random.shuffle(indices)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    if n == 1:
        train_idx = val_idx = [0]

    train_ds = VesuviusDataset3D([vol_paths[i] for i in train_idx], [lbl_paths[i] for i in train_idx],
                                  CFG.patch_size, CFG.stride, augment=True, backend="zarr")
    val_ds = VesuviusDataset3D([vol_paths[i] for i in val_idx], [lbl_paths[i] for i in val_idx],
                                CFG.patch_size, CFG.patch_size, augment=False, backend="zarr")

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True)

    module = SurfaceDetectionModule()

    checkpoint_cb = ModelCheckpoint(
        dirpath=CFG.output_dir, filename="segnext3d-{epoch}-{val_dice:.4f}",
        monitor="val_dice", mode="max", save_top_k=1,
    )
    early_stop_cb = EarlyStopping(monitor="val_dice", mode="max", patience=7)

    trainer = pl.Trainer(
        max_epochs=CFG.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if CFG.use_amp else 32,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=10,
        default_root_dir=CFG.output_dir,
    )
    trainer.fit(module, train_loader, val_loader)
    return module.model


if __name__ == "__main__":
    main_train()
