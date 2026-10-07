from pathlib import Path

import h5py
import numpy as np


def generate_darkcount_noise(dnl_matrix, histogram_matrixc, darkcount_map, noise_factor, seed=None):

    """
    Generate darkcount for a histogram cube.

    Parameters
    ----------
    dnl_matrix : ndarray or None
        3D DNL matrix with shape (W, H, B). Bin widths are treated as
        proportional to (1 + dnl). If None, bins are assumed evenly spaced.
    histogram_matrixc : ndarray
        Input photon histogram matrix with shape (W, H, B).
    darkcount_map : ndarray
        2D darkcount dark count with shape (W, H). noise strength per pixel is
        proportional to this map.
    noise_factor : float
        Noise strength in percent of total photon count in `histogram_matrixc`.
        Example: noise_factor=5 means noise photons ~= 5% of total photons.
    seed : int or None
        Optional RNG seed.

    Returns
    -------
    ambient_noise_matrix : ndarray
        Integer noise histogram with shape (W, H, B).
    """

    histogram_matrixc = np.asarray(histogram_matrixc)
    darkcount_map = np.asarray(darkcount_map)

    if histogram_matrixc.ndim != 3:
        raise ValueError("histogram_matrixc must be 3D with shape (W, H, B).")
    if darkcount_map.ndim != 2:
        raise ValueError("darkcount_map must be 2D with shape (W, H).")

    width, height, bin_number = histogram_matrixc.shape
    if darkcount_map.shape != (width, height):
        raise ValueError(
            f"Shape mismatch: darkcount_map{darkcount_map.shape} must match (W, H)=({width}, {height})."
        )
    if dnl_matrix is not None:
        dnl_matrix = np.asarray(dnl_matrix)
        if dnl_matrix.ndim != 3:
            raise ValueError("dnl_matrix must be 3D with shape (W, H, B), or None.")
        if dnl_matrix.shape != histogram_matrixc.shape:
            raise ValueError(
                f"Shape mismatch: dnl_matrix{dnl_matrix.shape} != histogram_matrixc{histogram_matrixc.shape}."
            )

    if float(noise_factor) < 0:
        raise ValueError("noise_factor must be >= 0.")

    total_photons = float(np.sum(histogram_matrixc))
    total_noise_photons = int(round(total_photons * (float(noise_factor) / 100.0)))

    ambient_noise_matrix = np.zeros((width, height, bin_number), dtype=np.int32)
    if total_noise_photons == 0:
        return ambient_noise_matrix

    rng = np.random.default_rng(seed)

    # Indirect ambient noise strength is proportional to darkcount value.
    pixel_darkcount = np.clip(darkcount_map.astype(np.float64), a_min=0.0, a_max=None)
    darkcount_sum = float(np.sum(pixel_darkcount))

    if darkcount_sum <= 0.0:
        pixel_prob = np.full(width * height, 1.0 / (width * height), dtype=np.float64)
    else:
        pixel_prob = (pixel_darkcount / darkcount_sum).reshape(-1)

    # Allocate total noise photons across pixels based on darkcount map.
    pixel_noise_counts = rng.multinomial(total_noise_photons, pixel_prob)

    # Bin probabilities from bin-width ratios:
    # p_k = width_k / sum(width), with width_k proportional to (1 + dnl_k).
    if dnl_matrix is None:
        bin_prob_matrix = np.full((width, height, bin_number), 1.0 / bin_number, dtype=np.float64)
    else:
        widths = 1.0 + dnl_matrix.astype(np.float64)
        widths = np.clip(widths, a_min=1e-12, a_max=None)
        widths_sum = np.sum(widths, axis=2, keepdims=True)
        invalid = widths_sum <= 0.0
        widths_sum[invalid] = float(bin_number)
        bin_prob_matrix = widths / widths_sum

    for pixel_idx, n_noise in enumerate(pixel_noise_counts):
        if n_noise == 0:
            continue

        i = pixel_idx // height
        j = pixel_idx % height
        ambient_noise_matrix[i, j, :] = rng.multinomial(n_noise, bin_prob_matrix[i, j, :])


    return ambient_noise_matrix



def read_darkcout_histogram(filename):
    with h5py.File(filename, 'r') as h5f:
        histogram = h5f["stamped_histogram"][:]
    return histogram


def _load_histogram_input(histogram_input):
    """Load one histogram from an H5 path or return an array input."""

    if isinstance(histogram_input, (str, bytes)) or hasattr(histogram_input, "__fspath__"):
        return np.asarray(read_darkcout_histogram(histogram_input))
    return np.asarray(histogram_input)


def generate_spatial_ratio_map(cover_test, experiment, save_location=None):
    """Generate a per-pixel early-peak ratio map from measured captures.

    The ratio is computed from the early temporal window (bins 9 through 13,
    inclusive):

    ``ratio[x, y] = experiment_early[x, y] / cover_test_early[x, y]``

    ``cover_test`` and ``experiment`` may be H5 paths or in-memory arrays.
    ``experiment`` may additionally be a sequence of paths/arrays; those
    captures are averaged before the ratio is calculated so the result remains
    a per-capture ratio.  Pixels with zero cover-test counts are assigned zero
    rather than producing NaN/Inf.

    Parameters
    ----------
    cover_test : path-like or ndarray
        Measured cover-test histogram with shape ``(W, H, B)``.
    experiment : path-like, ndarray, or sequence thereof
        One or more experiment histograms with the same shape as ``cover_test``.
    save_location : path-like or None, optional
        Optional output path.  HDF5 (``.h5``/``.hdf5``) is the preferred
        format and stores the map in the ``spatial_ratio_map`` dataset.  A
        legacy ``.npy`` suffix is also accepted.  If a directory (or
        suffix-less path) is supplied, ``spatial_ratio_mask.h5`` is written
        inside it.

    Returns
    -------
    ndarray
        ``float32`` ratio map with shape ``(W, H)``.
    """

    cover = _load_histogram_input(cover_test)
    if cover.ndim != 3:
        raise ValueError("cover_test must have shape (W, H, B).")

    if isinstance(experiment, (list, tuple)):
        experiment_captures = [_load_histogram_input(item) for item in experiment]
    else:
        experiment_captures = [_load_histogram_input(experiment)]

    if not experiment_captures:
        raise ValueError("experiment must contain at least one histogram.")

    expected_shape = cover.shape
    for capture in experiment_captures:
        if capture.ndim != 3:
            raise ValueError("Each experiment histogram must have shape (W, H, B).")
        if capture.shape != expected_shape:
            raise ValueError(
                "cover_test and experiment histograms must have identical shapes; "
                f"got {expected_shape} and {capture.shape}."
            )

    _, _, bin_number = expected_shape
    if bin_number < 14:
        raise ValueError(
            "The histogram must contain at least 14 bins to use early bins 9-13."
        )

    early_slice = slice(9, 14)
    cover_early = np.clip(cover.astype(np.float64), 0.0, None)[:, :, early_slice].sum(axis=2)
    experiment_early = np.zeros(expected_shape[:2], dtype=np.float64)
    for capture in experiment_captures:
        experiment_early += np.clip(
            capture.astype(np.float64), 0.0, None
        )[:, :, early_slice].sum(axis=2)
    experiment_early /= float(len(experiment_captures))

    ratio_map = np.divide(
        experiment_early,
        cover_early,
        out=np.zeros_like(experiment_early, dtype=np.float64),
        where=cover_early > 0.0,
    ).astype(np.float32)

    if save_location is not None:
        save_path = Path(save_location)
        if (save_path.exists() and save_path.is_dir()) or save_path.suffix.lower() not in {
            ".h5",
            ".hdf5",
            ".npy",
        }:
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / "spatial_ratio_mask.h5"
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix.lower() in {".h5", ".hdf5"}:
            with h5py.File(save_path, "w") as h5f:
                dataset = h5f.create_dataset(
                    "spatial_ratio_map",
                    data=ratio_map,
                    dtype=np.float32,
                )
                dataset.attrs["early_bin_start"] = 9
                dataset.attrs["early_bin_end_inclusive"] = 13
                dataset.attrs["zero_denominator_value"] = 0.0
                dataset.attrs["description"] = (
                    "Experiment early-peak counts divided by cover-test "
                    "early-peak counts per pixel."
                )
                h5f.attrs["map_type"] = "spatial_ratio_map"
                h5f.attrs["valid_pixel_count"] = int(np.count_nonzero(cover_early > 0.0))
                h5f.attrs["pixel_count"] = int(cover_early.size)
        else:
            np.save(save_path, ratio_map)

    return ratio_map


def read_spatial_ratio_map(filename):
    """Read a spatial ratio map from HDF5 or NumPy format."""

    path = Path(filename)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as h5f:
            if "spatial_ratio_map" not in h5f:
                raise KeyError(
                    f"{path} does not contain the 'spatial_ratio_map' dataset."
                )
            return np.asarray(h5f["spatial_ratio_map"][:])
    return np.asarray(np.load(path))


def add_darkCount_spatial_ratio_map(
    source_histogram,
    darkcount_histogram,
    spatial_ratio_map,
    percenter_value,
    seed=None,
    spatial_jitter_sigma=0.08,
):
    """Add Poisson-sampled dark counts using a measured spatial ratio map.

    The ratio map is normalized by its dark-count-weighted mean, so it changes
    the spatial distribution without changing the requested global
    ``percenter_value``.  The temporal-bin distribution remains the empirical
    distribution measured in ``darkcount_histogram``.  A log-normal spatial
    perturbation gives each capture a small, reproducible sensor variation.
    """

    source_histogram = np.asarray(source_histogram)
    darkcount_histogram = np.asarray(darkcount_histogram)
    spatial_ratio_map = np.asarray(spatial_ratio_map)

    if source_histogram.ndim != 3 or darkcount_histogram.ndim != 3:
        raise ValueError("source_histogram and darkcount_histogram must be 3D.")
    if source_histogram.shape != darkcount_histogram.shape:
        raise ValueError(
            "source_histogram and darkcount_histogram must have the same shape."
        )
    if spatial_ratio_map.shape != source_histogram.shape[:2]:
        raise ValueError(
            "spatial_ratio_map must match the spatial histogram shape "
            f"{source_histogram.shape[:2]}, got {spatial_ratio_map.shape}."
        )
    if float(percenter_value) < 0:
        raise ValueError("percenter_value must be >= 0.")
    if float(spatial_jitter_sigma) < 0:
        raise ValueError("spatial_jitter_sigma must be >= 0.")
    if float(percenter_value) == 0:
        return source_histogram.copy()

    reference = np.clip(darkcount_histogram.astype(np.float64), 0.0, None)
    pixel_totals = reference.sum(axis=2)
    reference_total = float(pixel_totals.sum())
    if reference_total <= 0.0:
        return source_histogram.copy()

    ratio = spatial_ratio_map.astype(np.float64)
    valid_ratio = np.isfinite(ratio) & (ratio > 0.0)
    ratio = np.where(valid_ratio, ratio, 1.0)

    weighted_ratio_mean = float(np.sum(pixel_totals * ratio) / reference_total)
    if not np.isfinite(weighted_ratio_mean) or weighted_ratio_mean <= 0.0:
        raise ValueError("spatial_ratio_map has no valid positive values.")
    normalized_ratio = ratio / weighted_ratio_mean

    rng = np.random.default_rng(seed)
    if float(spatial_jitter_sigma) > 0.0:
        jitter = np.exp(
            rng.normal(
                loc=-0.5 * float(spatial_jitter_sigma) ** 2,
                scale=float(spatial_jitter_sigma),
                size=normalized_ratio.shape,
            )
        )
        normalized_ratio *= jitter

    temporal_totals = reference.sum(axis=(0, 1))
    temporal_total = float(temporal_totals.sum())
    if temporal_total <= 0.0:
        temporal_prob = np.full(
            reference.shape[2], 1.0 / reference.shape[2], dtype=np.float64
        )
    else:
        temporal_prob = temporal_totals / temporal_total

    expected_counts = (
        pixel_totals[:, :, None]
        * normalized_ratio[:, :, None]
        * temporal_prob[None, None, :]
        * (float(percenter_value) / 100.0)
    )
    sampled_darkcount = rng.poisson(expected_counts).astype(np.int64, copy=False)
    combined_histogram = source_histogram.astype(np.int64) + sampled_darkcount

    if np.issubdtype(source_histogram.dtype, np.integer):
        return combined_histogram.astype(source_histogram.dtype)
    return combined_histogram.astype(source_histogram.dtype, copy=False)


# PEP 8 alias for callers that prefer snake_case; keep the requested public
# name above for compatibility with existing configuration terminology.
add_dark_count_spatial_ratio_map = add_darkCount_spatial_ratio_map


def sample_random_dark_histogram(darkcount_histogram, percenter_value, seed=None):
    """Sample pixel-wise dark noise from a measured sensor distribution.

    The reference histogram is used only to estimate two marginal
    distributions: the distribution of total dark counts across pixels and
    the global temporal-bin distribution.  Pixel strengths are randomly
    reassigned for every sample, then independent Poisson counts are drawn for
    every pixel/bin.  This preserves the measured sensor statistics without
    stamping the same spatial dark-count map onto every synthetic histogram.
    """

    darkcount_histogram = np.asarray(darkcount_histogram)
    if darkcount_histogram.ndim != 3:
        raise ValueError("darkcount_histogram must be 3D with shape (W, H, B).")
    if float(percenter_value) < 0:
        raise ValueError("percenter_value must be >= 0.")

    width, height, bin_number = darkcount_histogram.shape
    if float(percenter_value) == 0:
        return np.zeros((width, height, bin_number), dtype=np.int64)

    reference = np.clip(
        darkcount_histogram.astype(np.float64),
        a_min=0.0,
        a_max=None,
    )
    reference_total = float(reference.sum())
    if reference_total <= 0.0:
        return np.zeros((width, height, bin_number), dtype=np.int64)

    rng = np.random.default_rng(seed)

    # Keep the empirical distribution of pixel dark-count strengths, but not
    # their fixed locations in the measured sensor frame.
    pixel_totals = reference.sum(axis=2).reshape(-1)
    randomized_pixel_totals = rng.permutation(pixel_totals).reshape(width, height)

    temporal_totals = reference.sum(axis=(0, 1))
    temporal_total = float(temporal_totals.sum())
    if temporal_total <= 0.0:
        temporal_prob = np.full(bin_number, 1.0 / bin_number, dtype=np.float64)
    else:
        temporal_prob = temporal_totals / temporal_total

    expected_counts = (
        randomized_pixel_totals[:, :, None]
        * temporal_prob[None, None, :]
        * (float(percenter_value) / 100.0)
    )
    return rng.poisson(expected_counts).astype(np.int64, copy=False)


def add_random_dark_histogram(
    source_histogram,
    darkcount_histogram,
    percenter_value,
    seed=None,
):
    """Add a newly sampled pixel-wise dark-count realization."""

    source_histogram = np.asarray(source_histogram)
    darkcount_histogram = np.asarray(darkcount_histogram)
    if source_histogram.shape != darkcount_histogram.shape:
        raise ValueError(
            "source_histogram and darkcount_histogram must have the same shape."
        )

    sampled_darkcount = sample_random_dark_histogram(
        darkcount_histogram=darkcount_histogram,
        percenter_value=percenter_value,
        seed=seed,
    )
    combined_histogram = source_histogram.astype(np.int64) + sampled_darkcount

    if np.issubdtype(source_histogram.dtype, np.integer):
        return combined_histogram.astype(source_histogram.dtype)
    return combined_histogram.astype(source_histogram.dtype, copy=False)


def add_dark_histogram(source_histogram, darkcount_histogram, percenter_value):
    source_histogram = np.asarray(source_histogram)
    darkcount_histogram = np.asarray(darkcount_histogram)

    if source_histogram.shape != darkcount_histogram.shape:
        raise ValueError(
            "source_histogram and darkcount_histogram must have the same shape."
        )

    if float(percenter_value) < 0:
        raise ValueError("percenter_value must be >= 0.")

    scaled_darkcount = darkcount_histogram.astype(np.float64) * (
        float(percenter_value) / 100.0
    )
    combined_histogram = source_histogram.astype(np.float64) + scaled_darkcount

    if np.issubdtype(source_histogram.dtype, np.integer):
        return np.rint(combined_histogram).astype(source_histogram.dtype)

    return combined_histogram.astype(source_histogram.dtype, copy=False)


# Canonical public names; legacy aliases retain the same implementation.
read_dark_count_histogram = read_darkcout_histogram
