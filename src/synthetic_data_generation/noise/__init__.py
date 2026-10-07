"""Sensor noise generation primitives.

The public names keep compatibility with the source project while modules use
the new package namespace.
"""

from .ambient import create_ambient_timeStamp
from .dark_count import (
    add_darkCount_spatial_ratio_map,
    add_dark_count_spatial_ratio_map,
    add_dark_histogram,
    add_random_dark_histogram,
    sample_random_dark_histogram,
)
from .diffuse_map import add_percentage_noise, read_diffuse_from_h5
from .time_jitter import (
    add_zero_jitter,
    create_emmision_jitter,
    create_emission_jitter,
)
from .relaxation_delay import add_zero_relaxation_delay, create_relaxation_delay

__all__ = [
    "add_darkCount_spatial_ratio_map",
    "add_dark_count_spatial_ratio_map",
    "add_dark_histogram",
    "add_percentage_noise",
    "add_random_dark_histogram",
    "add_zero_jitter",
    "add_zero_relaxation_delay",
    "create_ambient_timeStamp",
    "create_emission_jitter",
    "create_emmision_jitter",
    "create_relaxation_delay",
    "read_diffuse_from_h5",
    "sample_random_dark_histogram",
]

from .ambient import create_ambient_timestamps
from .timestamps import read_pixelized_timestamps, merge_pixelized_timestamps, count_timestamps
from .dark_count import read_dark_count_histogram

__all__ += ["create_ambient_timestamps", "read_pixelized_timestamps", "merge_pixelized_timestamps", "count_timestamps", "read_dark_count_histogram"]
