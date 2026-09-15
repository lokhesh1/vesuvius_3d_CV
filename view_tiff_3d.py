"""Open a multi-page TIFF as an interactive 3D volume in a browser."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tifffile


def build_figure(volume: np.ndarray, surface_count: int) -> go.Figure:
    """Create a Plotly volume-rendering figure from a z, y, x array."""
    depth, height, width = volume.shape
    z, y, x = np.mgrid[
        0:depth:complex(depth),
        0:height:complex(height),
        0:width:complex(width),
    ]

    return go.Figure(
        go.Volume(
            x=x.ravel(),
            y=y.ravel(),
            z=z.ravel(),
            value=volume.ravel(),
            isomin=float(np.percentile(volume, 35)),
            isomax=float(np.percentile(volume, 99.5)),
            surface_count=surface_count,
            opacity=0.12,
            colorscale="Gray",
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title="Intensity"),
        )
    ).update_layout(
        title="Interactive 3D TIFF volume",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Slice",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tiff", type=Path, help="Path to a multi-page TIFF")
    parser.add_argument(
        "--step",
        type=int,
        default=3,
        help="Keep every Nth voxel in each direction (default: 3)",
    )
    parser.add_argument(
        "--surface-count",
        type=int,
        default=10,
        help="Number of intensity surfaces to render (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tiff_3d_view.html"),
        help="Output HTML file (default: tiff_3d_view.html)",
    )
    args = parser.parse_args()

    if args.step < 1:
        parser.error("--step must be at least 1")
    if args.surface_count < 1:
        parser.error("--surface-count must be at least 1")

    volume = tifffile.imread(args.tiff)
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a multi-page 3D TIFF, but received an array with shape "
            f"{volume.shape}."
        )

    # Downsampling keeps the generated HTML responsive while preserving the volume.
    volume = volume[:: args.step, :: args.step, :: args.step].astype(np.float32)
    low, high = np.percentile(volume, (1, 99.5))
    if high <= low:
        raise ValueError("The TIFF does not contain a usable intensity range.")
    volume = np.clip((volume - low) / (high - low), 0, 1)

    figure = build_figure(volume, args.surface_count)
    figure.write_html(args.output, include_plotlyjs=True)
    output = args.output.resolve()
    print(f"Loaded volume: {volume.shape[0]} slices x {volume.shape[1]} x {volume.shape[2]}")
    print(f"Saved interactive viewer: {output}")
    webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
