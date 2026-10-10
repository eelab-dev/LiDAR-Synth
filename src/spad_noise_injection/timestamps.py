from __future__ import annotations

from pathlib import Path
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np



PHOTON_DTYPE = np.dtype([
    ("time_ps", np.float32),
    ("collision_count", np.int32),
    ("emmision_delay_ps", np.float32),
    ("relaxation_delay_ps", np.float32),
])

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
PS_PER_SECOND = 1e12
MM_PER_METER = 1000.0

def is_empty_pixel(entries):
    if entries is None:
        return True
    try:
        return len(entries) == 0
    except TypeError:
        return False

def extract_collison_count(entry):
    if isinstance(entry, np.void) and entry.dtype.names:
        if "collision_count" not in entry.dtype.names:
            warnings.warn(
                "Timestamp entry is missing required 'collision_count' field; "
                "defaulting to 0.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0
    return int(entry["collision_count"])

def extract_time_ps(entry):
    if isinstance(entry, np.void) and entry.dtype.names:
        if "time_ps" not in entry.dtype.names:
            warnings.warn(
                "Timestamp entry is missing required 'time_ps' field; "
                "defaulting to 0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0
        return float(entry["time_ps"])
    return float(entry[0])

def extract_jitter_time_ps(entry):
    if isinstance(entry, np.void) and entry.dtype.names:
        if "emmision_delay_ps" not in entry.dtype.names:
            warnings.warn(
                "Timestamp entry is missing required 'emmision_delay_ps' field; "
                "defaulting to 0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0
        return float(entry["emmision_delay_ps"])
    return 0

def extract_relaxation_time_ps(entry):
    if isinstance(entry, np.void) and entry.dtype.names:
        if "relaxation_delay_ps" not in entry.dtype.names:
            warnings.warn(
                "Timestamp entry is missing required 'relaxation_delay_ps' field; "
                "defaulting to 0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0
        return float(entry["relaxation_delay_ps"])
    return 0

def _distance_mm_to_time_ps(distance_mm: np.ndarray) -> np.ndarray:
    """Convert one-way path length in millimetres to picoseconds."""
    distance_m = np.asarray(distance_mm, dtype=np.float64) / MM_PER_METER
    return (distance_m / 2 / SPEED_OF_LIGHT_M_PER_S * PS_PER_SECOND).astype(np.float32)


def _append_time_ps_field(pixel_entries: np.ndarray) -> np.ndarray:
    pixel_entries = np.asarray(pixel_entries)

    if pixel_entries.dtype.names is None:
        return pixel_entries

    if "distance" not in pixel_entries.dtype.names:
        if "time_ps" in pixel_entries.dtype.names:
            return pixel_entries
        return pixel_entries

    if "time_ps" in pixel_entries.dtype.names:
        updated_entries = pixel_entries.copy()
        updated_entries["time_ps"] = _distance_mm_to_time_ps(updated_entries["distance"])
        return updated_entries

    renamed_dtype = np.dtype([
        ("time_ps" if field_name == "distance" else field_name, pixel_entries.dtype.fields[field_name][0])
        for field_name in pixel_entries.dtype.names
    ])
    updated_entries = np.empty(pixel_entries.shape, dtype=renamed_dtype)
    for field_name in pixel_entries.dtype.names:
        target_field_name = "time_ps" if field_name == "distance" else field_name
        if field_name == "distance":
            updated_entries[target_field_name] = _distance_mm_to_time_ps(
                pixel_entries[field_name]
            )
        else:
            updated_entries[target_field_name] = pixel_entries[field_name]
    return updated_entries


def read_pixllized_timeStamp_from_h5(
    pixelized_timestamp: str
) -> np.ndarray:
    """Read photon_data and replace ``distance`` in mm with ``time_ps``."""
    file_path = Path(pixelized_timestamp)

    with h5py.File(file_path, "r") as h5f:
        if "photon_data" not in h5f:
            raise KeyError(f"'photon_data' dataset not found in {file_path}")

        photon_data = h5f["photon_data"][:]

    if photon_data.dtype == object:
        converted_photon_data = np.empty(photon_data.shape, dtype=object)
        for row in range(photon_data.shape[0]):
            for col in range(photon_data.shape[1]):
                converted_photon_data[row, col] = _append_time_ps_field(
                    photon_data[row, col]
                )
        return converted_photon_data

    photon_data = _append_time_ps_field(photon_data)
    return photon_data





def get_timeStamp_count_matrix(
    pixelized_timestamp: str
) -> np.ndarray:
    """Return the per-pixel timestamp count from a pixelized HDF5 file."""
    file_path = Path(pixelized_timestamp)

    with h5py.File(file_path, "r") as h5f:
        if "photon_data" not in h5f:
            raise KeyError(f"'photon_data' dataset not found in {file_path}")

        photon_data = h5f["photon_data"]
        timestamp_count = np.zeros(photon_data.shape, dtype=np.int32)

        for row in range(photon_data.shape[0]):
            for col in range(photon_data.shape[1]):
                timestamp_count[row, col] = len(photon_data[row, col])

    return timestamp_count


def get_timeStamp_count_sum(
    pixelized_timestamp: str
) -> int:
    """Return the total timestamp count from a pixelized HDF5 file."""
    timestamp_count = get_timeStamp_count_matrix(pixelized_timestamp)
    return int(timestamp_count.sum())


def get_timeStamp_count_sum_from_matrix(
    pixelized_timestamp: np.ndarray
) -> int:
    """Return the total timestamp count from a 2D pixelized timestamp matrix."""
    pixelized_timestamp = np.asarray(pixelized_timestamp, dtype=object)

    if pixelized_timestamp.ndim != 2:
        raise ValueError("pixelized_timestamp must be a 2D object array")

    total_timestamp_count = 0
    for row in range(pixelized_timestamp.shape[0]):
        for col in range(pixelized_timestamp.shape[1]):
            total_timestamp_count += len(pixelized_timestamp[row, col])

    return int(total_timestamp_count)


def extract_pixel_timeStamp(
    x_index: int,
    y_index: int,
    input_timeStamps: np.ndarray,
) -> np.ndarray:
    """Return a 1x1 pixelized timestamp view while preserving the outer structure."""
    input_timeStamps = np.asarray(input_timeStamps, dtype=object)

    if input_timeStamps.ndim != 2:
        raise ValueError("input_timeStamps must be a 2D pixelized timestamp array")

    image_height, image_width = input_timeStamps.shape
    if not 0 <= x_index < image_width:
        raise IndexError(f"x_index {x_index} is out of bounds for width {image_width}")
    if not 0 <= y_index < image_height:
        raise IndexError(f"y_index {y_index} is out of bounds for height {image_height}")

    output_timeStamps = np.empty((1, 1), dtype=object)
    output_timeStamps[0, 0] = input_timeStamps[y_index, x_index]
    return output_timeStamps


def display_pixelized_timeStamp_intensity(
    pixelized_timestamp: np.ndarray
) -> np.ndarray:
    """Display a per-pixel intensity image from a pixelized timestamp object array."""
    pixelized_timestamp = np.asarray(pixelized_timestamp, dtype=object)
    if pixelized_timestamp.ndim != 2:
        raise ValueError("pixelized_timestamp must be a 2D object array")

    intensity_image = np.zeros(pixelized_timestamp.shape, dtype=np.int32)

    for row in range(pixelized_timestamp.shape[0]):
        for col in range(pixelized_timestamp.shape[1]):
            intensity_image[row, col] = len(pixelized_timestamp[row, col])

    display_image = np.ma.masked_where(intensity_image <= 0, intensity_image)
    plt.figure()
    plt.imshow(display_image.T, cmap="viridis", interpolation="nearest", origin="lower")
    plt.colorbar(label="Photon Count")
    plt.title("Pixelized Timestamp Intensity")
    plt.show()

    return intensity_image


def merge_pixelized_timeStamp(
    pixelized_timestamp_1: np.ndarray,
    pixelized_timestamp_2: np.ndarray,
) -> np.ndarray:
    """Merge two pixelized timestamp object arrays pixel by pixel."""
    pixelized_timestamp_1 = np.asarray(pixelized_timestamp_1, dtype=object)
    pixelized_timestamp_2 = np.asarray(pixelized_timestamp_2, dtype=object)

    if pixelized_timestamp_1.ndim != 2 or pixelized_timestamp_2.ndim != 2:
        raise ValueError("Both inputs must be 2D object arrays")
    if pixelized_timestamp_1.shape != pixelized_timestamp_2.shape:
        raise ValueError("Both inputs must have the same shape")

    merged_pixelized_timestamp = np.empty(pixelized_timestamp_1.shape, dtype=object)

    for row in range(pixelized_timestamp_1.shape[0]):
        for col in range(pixelized_timestamp_1.shape[1]):
            pixel_1 = np.asarray(pixelized_timestamp_1[row, col])
            pixel_2 = np.asarray(pixelized_timestamp_2[row, col])

            if pixel_1.dtype != pixel_2.dtype:
                raise ValueError(
                    f"Pixel dtype mismatch at ({row}, {col}): "
                    f"{pixel_1.dtype} != {pixel_2.dtype}"
                )

            merged_pixel = np.concatenate((pixel_1, pixel_2))
            if merged_pixel.size > 0 and merged_pixel.dtype.names:
                if "time_ps" in merged_pixel.dtype.names:
                    order = np.argsort(merged_pixel["time_ps"])
                    merged_pixel = merged_pixel[order]
                elif "distance" in merged_pixel.dtype.names:
                    raise ValueError(
                        "need to convert to time of flight"
                )


            merged_pixelized_timestamp[row, col] = merged_pixel

    return merged_pixelized_timestamp


def generate_normal_distribution_timeStamp(
    mean_time_ps: float,
    sigma: float,
    photon_count: int,
) -> np.ndarray:
    """
    Generate a normal distribution of timestamps.

    Args:
        mean_time_ps: Mean time in picoseconds
        sigma: Standard deviation in picoseconds
        photon_count: Number of photons to generate

    Returns:
        np.ndarray: Array of timestamps with PHOTON_DTYPE containing normally distributed time_ps values
    """
    if photon_count < 0:
        raise ValueError("photon_count must be non-negative")
    if sigma < 0:
        raise ValueError("sigma (standard deviation) must be non-negative")

    if photon_count == 0:
        return np.array([], dtype=PHOTON_DTYPE)

    # Generate normally distributed timestamps
    timestamps = np.random.normal(mean_time_ps, sigma, photon_count)

    # Create structured array with PHOTON_DTYPE
    photon_data = np.empty(photon_count, dtype=PHOTON_DTYPE)
    photon_data["time_ps"] = timestamps.astype(np.float32)
    photon_data["collision_count"] = np.zeros(photon_count, dtype=np.int32)
    photon_data["emmision_delay_ps"] = np.zeros(photon_count, dtype=np.float32)
    photon_data["relaxation_delay_ps"] = np.zeros(photon_count, dtype=np.float32)

    return photon_data


# Canonical public names; legacy aliases retain the same implementation.
read_pixelized_timestamps = read_pixllized_timeStamp_from_h5
merge_pixelized_timestamps = merge_pixelized_timeStamp
count_timestamps = get_timeStamp_count_sum
