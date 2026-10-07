
from __future__ import annotations

import h5py
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

depthmap_colors = np.array([
    [0.55, 0.0, 0.0, 1.0],  # Dark Red
    [0.85, 0.0, 0.0, 1.0],  # Crimson
    [1.0, 0.0, 0.0, 1.0],   # Red
    [1.0, 0.3, 0.0, 1.0],   # Vermilion
    [1.0, 0.55, 0.0, 1.0],  # Dark Orange
    [1.0, 0.65, 0.0, 1.0],  # Orange
    [1.0, 0.84, 0.0, 1.0],  # Gold
    [1.0, 1.0, 0.0, 1.0],   # Yellow
    [0.8, 1.0, 0.2, 1.0],   # Yellow-Green
    [0.5, 1.0, 0.0, 1.0],   # Light Green
    [0.0, 1.0, 0.0, 1.0],   # Green
    [0.0, 1.0, 0.5, 1.0],   # Spring Green
    [0.0, 0.75, 0.5, 1.0],  # Turquoise
    [0.0, 1.0, 1.0, 1.0],   # Cyan
    [0.0, 0.5, 1.0, 1.0],   # Sky Blue
    [0.0, 0.0, 1.0, 1.0],   # Blue
    [0.0, 0.0, 0.5, 1.0],   # Dark Blue
], dtype=np.float32)

# Normalize the colors for LinearSegmentedColormap
positions = np.linspace(0, 1, depthmap_colors.shape[0])
colors = [(pos, color) for pos, color in zip(positions, depthmap_colors)]


def display_depth_image(
    depth: np.ndarray,
    title: str = "Depth Image",
    invalid_value: float = -1.0,
    ax=None,
):
    """
    Display a depth image using the globally defined depth colormap.

    depth: 2D depth image.
    title: Title for the plot.
    invalid_value: Pixels with this value are treated as invalid.
    ax: Optional matplotlib axis to draw on.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array.")
    if depth.size == 0:
        raise ValueError("depth must not be empty.")

    masked_depth = np.ma.masked_where(depth == invalid_value, depth)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "depthmap_colormap", colors
    ).copy()
    cmap.set_bad(color=(0.0, 0.0, 0.0, 1.0))

    if ax is None:
        _, ax = plt.subplots()

    image = ax.imshow(masked_depth, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(image, ax=ax, label="Depth")

    if ax.figure.canvas is not None:
        ax.figure.tight_layout()

    return ax


def save_depth_image_png(
    depth: np.ndarray,
    title: str = "Depth Image",
    invalid_value: float = -1.0,
    ax=None,
    *,
    save_path,
    range_max: float | None = None,
):
    """
    Render a depth image in the histogram-viewer style and save it as a PNG.

    depth: 2D depth image.
    title: Title for the plot.
    invalid_value: Pixels with this value are treated as invalid.
    ax: Optional matplotlib axis to draw on.
    save_path: Destination path for the PNG image.
    range_max: Fixed maximum bin index for the color scale. When omitted,
        the maximum valid depth in this image is used.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array.")
    if depth.size == 0:
        raise ValueError("depth must not be empty.")

    masked_depth = np.ma.masked_where(depth == invalid_value, depth)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "depth_cmap", colors, N=256
    ).copy()
    cmap.set_bad(color="black")

    if ax is None:
        _, ax = plt.subplots()

    if range_max is None:
        valid_depth = masked_depth.compressed()
        range_max = float(valid_depth.max() + 1) if valid_depth.size else 1.0
    elif range_max <= 0:
        raise ValueError("range_max must be greater than 0.")

    image = ax.imshow(
        masked_depth.T,
        cmap=cmap,
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=range_max,
    )
    ax.set_title(title)
    plt.colorbar(image, ax=ax)

    ax.figure.savefig(save_path, format="png")
    return ax


def form_depth_from_histogram(
    histogram: np.ndarray, peak_func=np.argmax
) -> np.ndarray:
    """
    Form depth estimates from a 3D histogram tensor by iterating over pixels
    and applying the provided peak function to each pixel histogram.

    histogram: 3D array where the last axis stores histogram-bin counts.
    peak_func: Function used to extract the depth/peak index from each 1D
        histogram.
    """
    histogram = np.asarray(histogram)
    if histogram.ndim != 3:
        raise ValueError("histogram must be a 3D array.")
    if histogram.size == 0:
        raise ValueError("histogram must not be empty.")
    if histogram.shape[-1] == 0:
        raise ValueError("histogram bin axis must not be empty.")

    depth = np.full(histogram.shape[:2], -1.0, dtype=float)
    for row in range(histogram.shape[0]):
        for col in range(histogram.shape[1]):
            pixel_histogram = histogram[row, col, :]
            depth[row, col] = float(peak_func(pixel_histogram))

    return depth

def peak_found_center_of_mass(bins: np.ndarray,peak_range: int = 8) -> float:

    total = bins.sum()
    if total <= 0:
        # no photons in this pixel, fall back to 0 (or any convention you like)
        return 0

    mid = np.argmax(bins)
    upper = peak_range // 2 + mid + 1
    low = upper - peak_range
    bin_num = bins.shape[0]

    low = max(0,low)
    upper = min(bin_num - 1,upper)

    indices = np.array(range(low,upper))
    total = 0.0
    com = 0.0
    for i in indices:
       com = i * bins[i] + com
       total = total + bins[i]
    try:
        if total <= 0:
            raise ValueError("Empty peak window: total weight is zero.")
        idx = com / total
        if not np.isfinite(idx):
            raise FloatingPointError("Center-of-mass result is not finite.")
    except (ValueError, ZeroDivisionError, FloatingPointError):
        # Fallback to the strongest-bin index when COM cannot be computed safely.
        return float(mid)

    if idx < 0 or idx >= bins.shape[0] -1:
        return -1

    # make sure it's within [0, len(bins)-1]
    return idx


def center_of_mass(bins: np.ndarray) -> float:
    """
    Return the index of the histogram's center of mass.

    bins: 1D array of counts for each bin.
    """
    total = bins.sum()
    if total <= 0:
        # no photons in this pixel, fall back to 0 (or any convention you like)
        return 0

    indices = np.arange(bins.shape[0], dtype=float)
    com = (indices * bins).sum() / total   # weighted average of indices

    # round to nearest integer bin index
    idx = com
    if idx < 0 or idx >= bins.shape[0] -1:
        return -1

    # make sure it's within [0, len(bins)-1]
    return idx


# peak_func=np.argmax
