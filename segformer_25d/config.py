import torch


class CFG:
    data_dir = "/kaggle/input/vesuvius-challenge-surface-detection"
    output_dir = "/kaggle/working"
    seed = 42
    patch_size = 128
    stride = 64
    batch_size = 2
    num_epochs = 25
    lr = 5e-4
    weight_decay = 1e-4
    num_workers = 2
    val_ratio = 0.2
    context_slices = 5
    slice_chunk = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True
