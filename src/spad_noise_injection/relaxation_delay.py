from __future__ import annotations

import numpy as np


RELAXATION_DELAY_FIELD = "relaxation_delay_ps"


def _append_relaxation_delay_field(
    pixel_entries: np.ndarray,
    rng: np.random.Generator,
    relaxation_tau: float,
) -> np.ndarray:
    pixel_entries = np.asarray(pixel_entries)

    if pixel_entries.dtype.names is None:
        raise ValueError("time_stamp entries must be structured arrays")

    relaxation_samples = rng.exponential(
        scale=relaxation_tau,
        size=pixel_entries.shape,
    ).astype(np.float32)

    if "collision_count" in pixel_entries.dtype.names:
        relaxation_samples[pixel_entries["collision_count"] < 0] = 0.0

    if RELAXATION_DELAY_FIELD in pixel_entries.dtype.names:
        updated_entries = pixel_entries.copy()
        updated_entries[RELAXATION_DELAY_FIELD] = relaxation_samples
        return updated_entries

    updated_dtype = np.dtype(
        list(pixel_entries.dtype.descr) + [(RELAXATION_DELAY_FIELD, np.float32)]
    )
    updated_entries = np.empty(pixel_entries.shape, dtype=updated_dtype)

    for field_name in pixel_entries.dtype.names:
        updated_entries[field_name] = pixel_entries[field_name]

    updated_entries[RELAXATION_DELAY_FIELD] = relaxation_samples
    return updated_entries


def create_relaxation_delay(
    time_stamp,
    seeds_number: int,
    relaxation_tau: float,
) -> np.ndarray:
    """
    Add detector relaxation-delay field in picoseconds.

    The delay model is:

        rng.exponential(scale=relaxation_tau, size=pixel_entries.shape)

    Parameters
    ----------
    time_stamp
        Pixelized timestamp data. This can be a 2D object array where each pixel
        contains a structured NumPy array, or a structured NumPy array directly.
    seeds_number
        Seed used to initialize the NumPy random generator.
    relaxation_tau
        Exponential scale parameter in picoseconds.
    """
    if relaxation_tau < 0:
        raise ValueError("relaxation_tau must be non-negative")

    rng = np.random.default_rng(seeds_number)
    time_stamp = np.asarray(time_stamp)

    if time_stamp.dtype == object:
        if time_stamp.ndim != 2:
            raise ValueError("time_stamp must be a 2D object array")

        delayed_time_stamp = np.empty(time_stamp.shape, dtype=object)

        for row in range(time_stamp.shape[0]):
            for col in range(time_stamp.shape[1]):
                delayed_time_stamp[row, col] = _append_relaxation_delay_field(
                    time_stamp[row, col],
                    rng,
                    relaxation_tau,
                )

        return delayed_time_stamp

    return _append_relaxation_delay_field(
        time_stamp,
        rng,
        relaxation_tau,
    )


def add_zero_relaxation_delay(time_stamp) -> np.ndarray:
    """
    Add or overwrite the relaxation-delay field with zeros in picoseconds.

    Parameters
    ----------
    time_stamp
        Pixelized timestamp data. This can be a 2D object array where each pixel
        contains a structured NumPy array, or a structured NumPy array directly.
    """
    return create_relaxation_delay(
        time_stamp,
        seeds_number=0,
        relaxation_tau=0.0,
    )
