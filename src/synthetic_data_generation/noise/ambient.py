from __future__ import annotations
from .timestamps import *
import numpy as np
from .timestamps import PHOTON_DTYPE



def _allocate_timestamp_counts(
    diffuse_map: np.ndarray,
    num_timestamps: int,
) -> np.ndarray:
    """Allocate integer timestamp counts per pixel following the diffuse map."""
    if num_timestamps < 0:
        raise ValueError("num_timestamps must be non-negative")

    flat_map = np.asarray(diffuse_map, dtype=np.float64)
    if flat_map.ndim != 2:
        raise ValueError("diffuse_map must be a 2D matrix")

    if flat_map.size == 0 or num_timestamps == 0:
        return np.zeros(flat_map.shape, dtype=np.int32)

    flat_weights = np.clip(flat_map, a_min=0.0, a_max=None).ravel()
    weight_sum = flat_weights.sum()

    if weight_sum <= 0:
        probabilities = np.full(flat_weights.shape, 1.0 / flat_weights.size, dtype=np.float64)
    else:
        probabilities = flat_weights / weight_sum

    expected_counts = probabilities * num_timestamps
    counts = np.floor(expected_counts).astype(np.int32)

    remainder = int(num_timestamps - counts.sum())
    if remainder > 0:
        fractional = expected_counts - counts
        top_indices = np.argsort(fractional)[-remainder:]
        counts[top_indices] += 1

    return counts.reshape(flat_map.shape)


def create_ambient_timeStamp(
    diffuse_map_matrix,
    min_range: float,
    max_range: float,
    num_timestamps: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Create per-pixel ambient timestamps with white-noise time distribution.

    The number of timestamps assigned to each pixel approximately follows the
    intensity of ``diffuse_map_matrix`` while each pixel's photon times are drawn
    uniformly between ``min_range`` and ``max_range``.

    The returned object matches the pixelized photon-data layout: a 2D object
    array where each pixel contains a structured NumPy array with fields
    ``time_ps`` and ``collision_count``. Ambient photons use
    ``collision_count = -1``.
    """
    diffuse_map = np.asarray(diffuse_map_matrix, dtype=np.float64)

    if diffuse_map.ndim != 2:
        raise ValueError("diffuse_map_matrix must be a 2D matrix")
    if max_range < min_range:
        raise ValueError("max_range must be greater than or equal to min_range")

    counts = _allocate_timestamp_counts(diffuse_map, num_timestamps)
    rng = np.random.default_rng(random_seed)

    ambient_timestamps = np.empty(counts.shape, dtype=object)
    for row in range(counts.shape[0]):
        for col in range(counts.shape[1]):
            pixel_count = int(counts[row, col])
            pixel_times_ps = np.sort(
                rng.uniform(min_range, max_range, size=pixel_count)
            ).astype(np.float32)

            pixel_timestamps = np.empty(pixel_count, dtype=PHOTON_DTYPE)
            pixel_timestamps["time_ps"] = pixel_times_ps
            pixel_timestamps["collision_count"] = -1
            pixel_timestamps["emmision_delay_ps"] = 0
            pixel_timestamps["relaxation_delay_ps"] = 0
            ambient_timestamps[row, col] = pixel_timestamps

    return ambient_timestamps


# Canonical public names; legacy aliases retain the same implementation.
create_ambient_timestamps = create_ambient_timeStamp
