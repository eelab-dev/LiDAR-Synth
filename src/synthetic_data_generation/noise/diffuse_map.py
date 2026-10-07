from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_diffuse_from_h5(
    diffuse_h5_file: str
) -> np.ndarray:
    """Read and return the diffuse_avg matrix from a diffuse-map HDF5 file."""
    file_path = Path(diffuse_h5_file)

    with h5py.File(file_path, "r") as h5f:
        if "diffuse_data" not in h5f:
            raise KeyError(f"'diffuse_data' dataset not found in {file_path}")

        diffuse_data = h5f["diffuse_data"][:]

    if "diffuse_avg" not in diffuse_data.dtype.names:
        raise KeyError(f"'diffuse_avg' field not found in diffuse_data for {file_path}")

    return diffuse_data["diffuse_avg"].astype(np.float32)


def add_percentage_noise(
    percentage_noise: float,
    diffuse_map: np.ndarray,
    random_seed: int | None = None,
) -> np.ndarray:
    """Return a diffuse map with random per-pixel noise scaled by a percentage.

    Each pixel is perturbed independently by a random factor sampled uniformly
    from ``[-percentage_noise, percentage_noise] / 100`` of its original value.
    Negative results are clipped to zero.
    """
    if percentage_noise < 0:
        raise ValueError("percentage_noise must be non-negative")

    diffuse_map_array = np.asarray(diffuse_map, dtype=np.float32)
    if diffuse_map_array.ndim != 2:
        raise ValueError("diffuse_map must be a 2D matrix")

    if diffuse_map_array.size == 0 or percentage_noise == 0:
        return diffuse_map_array.copy()

    rng = np.random.default_rng(random_seed)
    noise_scale = percentage_noise / 100.0
    noise = rng.uniform(
        low=-noise_scale,
        high=noise_scale,
        size=diffuse_map_array.shape,
    ).astype(np.float32)

    noisy_diffuse_map = diffuse_map_array * (1.0 + noise)
    return np.clip(noisy_diffuse_map, a_min=0.0, a_max=None).astype(np.float32)


def display_diffuse_map(diffuse_map: np.ndarray) -> None:
    """Display a diffuse map returned by ``read_diffuse_from_h5``."""
    diffuse_map_array = np.asarray(diffuse_map, dtype=np.float32)
    if diffuse_map_array.ndim != 2:
        raise ValueError("diffuse_map must be a 2D matrix")

    plt.figure()
    plt.imshow(diffuse_map_array.T, cmap="viridis", interpolation='nearest', origin='lower')
    plt.colorbar(label="Diffuse intensity")
    plt.title("Diffuse Map")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()
