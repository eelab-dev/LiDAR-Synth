from __future__ import annotations

import numpy as np


EMMISION_JITTER_FIELD = "emmision_delay_ps"


def _append_emmision_jitter_field(
    pixel_entries: np.ndarray,
    rng: np.random.Generator,
    jitter_sigma: float,
    emission_mean = 0,
) -> np.ndarray:
    pixel_entries = np.asarray(pixel_entries)

    if pixel_entries.dtype.names is None:
        raise ValueError("time_stamp entries must be structured arrays")

    # Main Gaussian jitter
    jitter_samples = rng.normal(
        loc=emission_mean,
        scale=jitter_sigma,
        size=pixel_entries.shape,
    ).astype(np.float32)

    if "collision_count" in pixel_entries.dtype.names:
        jitter_samples[pixel_entries["collision_count"] < 0] = 0.0

    if EMMISION_JITTER_FIELD in pixel_entries.dtype.names:
        updated_entries = pixel_entries.copy()
        updated_entries[EMMISION_JITTER_FIELD] = jitter_samples
        return updated_entries

    updated_dtype = np.dtype(
        list(pixel_entries.dtype.descr) + [(EMMISION_JITTER_FIELD, np.float32)]
    )
    updated_entries = np.empty(pixel_entries.shape, dtype=updated_dtype)

    for field_name in pixel_entries.dtype.names:
        updated_entries[field_name] = pixel_entries[field_name]

    updated_entries[EMMISION_JITTER_FIELD] = jitter_samples
    return updated_entries



def create_emmision_jitter(
    time_stamp,
    seeds_number: int,
    jitter_sigma: float,
    emission_mean=0,
) -> np.ndarray:
    """
    Add emission-delay field in picoseconds.

    The jitter model is:

        Gaussian jitter + optional positive exponential tail

    Parameters
    ----------
    jitter_sigma
        Gaussian jitter standard deviation in ps.

    """
    if jitter_sigma < 0:
        raise ValueError("jitter_sigma must be non-negative")


    rng = np.random.default_rng(seeds_number)
    time_stamp = np.asarray(time_stamp)

    if time_stamp.dtype == object:
        if time_stamp.ndim != 2:
            raise ValueError("time_stamp must be a 2D object array")

        jittered_time_stamp = np.empty(time_stamp.shape, dtype=object)

        for row in range(time_stamp.shape[0]):
            for col in range(time_stamp.shape[1]):
                jittered_time_stamp[row, col] = _append_emmision_jitter_field(
                    time_stamp[row, col],
                    rng,
                    jitter_sigma,
                    emission_mean,
                )

        return jittered_time_stamp

    return _append_emmision_jitter_field(
        time_stamp,
        rng,
        jitter_sigma,
        emission_mean,
    )


create_emission_jitter = create_emmision_jitter


def add_zero_jitter(time_stamp,emission_delay = 0) -> np.ndarray:
    """
    Add or overwrite the emission-delay field with a constant value in picoseconds.

    Parameters
    ----------
    time_stamp
        Pixelized timestamp data. This can be a 2D object array where each pixel
        contains a structured NumPy array, or a structured NumPy array directly.
    emission_delay
        Constant value written to the emission-delay field in ps.
    """
    if emission_delay < 0:
        raise ValueError("emission_delay must be non-negative")

    time_stamp = np.asarray(time_stamp)
    rng = np.random.default_rng(0)

    if time_stamp.dtype == object:
        if time_stamp.ndim != 2:
            raise ValueError("time_stamp must be a 2D object array")

        zero_jitter_time_stamp = np.empty(time_stamp.shape, dtype=object)
        for row in range(time_stamp.shape[0]):
            for col in range(time_stamp.shape[1]):
                zero_jitter_time_stamp[row, col] = _append_emmision_jitter_field(
                    time_stamp[row, col],
                    rng,
                    0.0,
                    emission_delay,
                )
        return zero_jitter_time_stamp

    return _append_emmision_jitter_field(time_stamp, rng, 0.0, emission_delay)
