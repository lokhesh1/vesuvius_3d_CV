import torch


class CFG:
    data_dir = "/kaggle/input/vesuvius-challenge-surface-detection"
    output_dir = "/kaggle/working"
    data_backend = "tif"  # "tif" for small tests, "zarr" for large datasets
    seed = 42
    patch_size = 128
    stride = 64
    batch_size = 2
    num_epochs = 30
    lr = 1e-3
    weight_decay = 1e-4
    num_workers = 0
    val_ratio = 0.2
    features = [32, 64, 128, 256]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True
