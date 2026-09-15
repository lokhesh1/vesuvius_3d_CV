"""Convert paired TIFF volumes to chunked Zarr arrays.

Usage:
    python -m common.convert_tif_to_zarr SOURCE_DIR DEST_DIR
"""

import argparse
from pathlib import Path

import tifffile
import zarr


def convert_split(source_dir, dest_dir, split, chunks, require_labels):
    image_dir = source_dir / f"{split}_images"
    label_dir = source_dir / f"{split}_labels"
    if not image_dir.exists():
        return

    out_image_dir = dest_dir / f"{split}_images"
    out_label_dir = dest_dir / f"{split}_labels"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.tif"))
    for image_path in image_paths:
        label_path = label_dir / image_path.name
        if require_labels and not label_path.exists():
            print(f"Skipping {image_path.name}: matching label not found")
            continue

        image = tifffile.imread(str(image_path))
        if image.ndim != 3:
            raise ValueError(f"Expected a 3D TIFF: {image_path}")
        label = None
        if require_labels:
            label = tifffile.imread(str(label_path))
            if image.shape != label.shape:
                raise ValueError(
                    f"Shape mismatch for {image_path.name}: "
                    f"{image.shape} vs {label.shape}"
                )

        image_out = out_image_dir / f"{image_path.stem}.zarr"
        image_store = zarr.open(
            str(image_out),
            mode="w",
            shape=image.shape,
            dtype=image.dtype,
            chunks=chunks,
        )
        image_store[:] = image
        if label is not None:
            label_out = out_label_dir / f"{image_path.stem}.zarr"
            label_store = zarr.open(
                str(label_out),
                mode="w",
                shape=label.shape,
                dtype=label.dtype,
                chunks=chunks,
            )
            label_store[:] = label
        print(f"Converted {split}/{image_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("dest_dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    chunks = (args.chunk_size,) * 3
    args.dest_dir.mkdir(parents=True, exist_ok=True)
    convert_split(args.source_dir, args.dest_dir, "train", chunks, require_labels=True)
    convert_split(args.source_dir, args.dest_dir, "test", chunks, require_labels=False)


if __name__ == "__main__":
    main()
