import os
import importlib
import warnings
from concurrent.futures import ThreadPoolExecutor

import h5py
import matplotlib.pyplot as plt
import numpy as np

from .timestamps import (
    extract_collison_count,
    extract_jitter_time_ps,
    extract_time_ps,
    extract_relaxation_time_ps,
)

try:
    from _synthetic_sensor import form_histogram_image as rust_form_histogram_image
except ImportError:
    rust_form_histogram_image = None

_RUST_HISTOGRAM_WARNING_EMITTED = False


def _load_rust_form_histogram_image():
    global rust_form_histogram_image

    if rust_form_histogram_image is not None:
        return rust_form_histogram_image

    import_errors = []
    module_specs = (
        ("_synthetic_sensor", "form_histogram_image"),
    )

    for module_name, attr_name in module_specs:
        try:
            module = importlib.import_module(module_name)
            rust_form_histogram_image = getattr(module, attr_name)
            return rust_form_histogram_image
        except Exception as exc:
            import_errors.append(f"{module_name}: {exc}")

    raise ImportError(
        "_synthetic_sensor is not available. Tried imports: "
        + " | ".join(import_errors)
    )




def compute_bin_edges(range_min, range_max, bin_number, dnl_row):
    """
    Compute bin edges from DNL for a single pixel.

    dnl_row: shape (B,)
    """
    ideal_width = (range_max - range_min) / bin_number
    widths = ideal_width * (1 + dnl_row)

    edges = np.zeros(bin_number + 1)
    edges[0] = range_min
    edges[1:] = range_min + np.cumsum(widths)

    return edges







def _process_histogram_pixel(
    y,
    x,
    entries,
    bin_number,
    range_min,
    range_max,
    ideal_edges,
    dnl_row=None,
):
    if dnl_row is None:
        edges = ideal_edges
    else:
        edges = compute_bin_edges(range_min, range_max, bin_number, dnl_row)

    illegal_entries = []
    stamped_pixel = np.zeros(bin_number, dtype=int)
    error_pixel = np.zeros(bin_number, dtype=int)
    multi_path_pixel = np.zeros(bin_number, dtype=int)
    true_pixel = np.zeros(bin_number, dtype=int)

    photon_number = len(entries)
    for k in range(photon_number):
        entry = entries[k]
        distance = extract_time_ps(entry)
        collosion = extract_collison_count(entry)
        jitter_time = extract_jitter_time_ps(entry)
        relaxation_time = extract_relaxation_time_ps(entry)

        distance += jitter_time / 2
        distance += relaxation_time / 2

        if range_min > distance or range_max < distance:
            illegal_entries.append((distance, collosion))
            continue

        bin_index = np.searchsorted(edges, distance) - 1
        bin_index = int(np.clip(bin_index, 0, bin_number - 1))
        stamped_pixel[bin_index] += 1

        if collosion < 0:
            error_pixel[bin_index] += 1

        if collosion > 2:
            multi_path_pixel[bin_index] += 1

        if collosion == 2:
            true_pixel[bin_index] += 1

    return y, x, illegal_entries, error_pixel, multi_path_pixel, stamped_pixel, true_pixel


def form_histogram_image(
    pixels,
    bin_number=25,
    range_distance=[1000, 2500],
    peak_func=np.argmax,
    dnl_matrix=None,
    num_workers=64,
):

    image_width, image_height = pixels.shape
    illegal_photon = np.empty((image_width,image_height), dtype=object)
    for i, j in np.ndindex(image_width,image_height):
        illegal_photon[i, j] = []

    stamped_histogram = np.zeros((image_width,image_height,bin_number), dtype=int)
    error_histogram = np.zeros((image_width,image_height,bin_number), dtype=int)
    multi_path_histogram =np.zeros((image_width,image_height,bin_number), dtype=int)
    true_histogram = np.zeros((image_width,image_height,bin_number), dtype=int)

    # distance_image = np.zeros((image_width,image_height), dtype=np.float32)
    range_min = range_distance[0]
    range_max = range_distance[1]
    if dnl_matrix is None:
        ideal_edges = np.linspace(range_min,range_max,bin_number+1)
    else:
        ideal_edges = None

    if num_workers is None:
        num_workers = min(32, os.cpu_count() or 1)

    tasks = []
    for y, row in enumerate(pixels):
        for x, entries in enumerate(row):
            tasks.append(
                (
                    y,
                    x,
                    entries,
                    bin_number,
                    range_min,
                    range_max,
                    ideal_edges,
                    None if dnl_matrix is None else dnl_matrix[y, x],
                )
            )

    if num_workers <= 1 or len(tasks) <= 1:
        pixel_results = (_process_histogram_pixel(*task) for task in tasks)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pixel_results = executor.map(
                lambda task: _process_histogram_pixel(*task),
                tasks,
            )
            pixel_results = list(pixel_results)

    for y, x, illegal_entries, error_pixel, multi_path_pixel, stamped_pixel, true_pixel in pixel_results:
        illegal_photon[y, x] = illegal_entries
        error_histogram[y, x, :] = error_pixel
        multi_path_histogram[y, x, :] = multi_path_pixel
        stamped_histogram[y, x, :] = stamped_pixel
        true_histogram[y, x, :] = true_pixel

    return  (illegal_photon,error_histogram,multi_path_histogram), stamped_histogram, true_histogram


def form_histogram_image_rust(
    pixels,
    bin_number=25,
    range_distance=[1000, 2500],
    peak_func=np.argmax,
    dnl_matrix=None,
    num_workers=64,
):
    global _RUST_HISTOGRAM_WARNING_EMITTED

    try:
        rust_impl = _load_rust_form_histogram_image()
    except ImportError as exc:
        if not _RUST_HISTOGRAM_WARNING_EMITTED:
            warnings.warn(
                "Rust histogram generation is unavailable; falling back to the "
                f"Python implementation. Details: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _RUST_HISTOGRAM_WARNING_EMITTED = True
        return form_histogram_image(
            pixels=pixels,
            bin_number=bin_number,
            range_distance=range_distance,
            peak_func=peak_func,
            dnl_matrix=dnl_matrix,
            num_workers=num_workers,
        )

    return rust_impl(
        pixels=pixels,
        bin_number=bin_number,
        range_distance=range_distance,
        peak_func=peak_func,
        dnl_matrix=dnl_matrix,
        num_workers=num_workers,
    )


def display_histogram_intensity_image(histogram: np.ndarray, max_intensity=None, ax=None, show_colorbar=True):
    histogram = np.asarray(histogram)
    if histogram.ndim != 3:
        raise ValueError("histogram must be a 3D array with shape (height, width, bins)")

    intensity_image = histogram.sum(axis=2)
    if max_intensity is not None:
        intensity_image = np.minimum(intensity_image, max_intensity)
    display_image = np.ma.masked_where(intensity_image <= 0, intensity_image)

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots()
        show_plot = True
    else:
        fig = ax.figure

    vmax = max_intensity if max_intensity is not None else intensity_image.max()
    image = ax.imshow(
        display_image.T,
        cmap="viridis",
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=vmax,
    )
    if show_colorbar:
        fig.colorbar(image, ax=ax, label="Photon Count")
    # ax.set_title("Histogram Intensity Image")

    if show_plot:
        plt.show()
    return ax




def save_histogram_to_h5(filename, stamped_histogram):
    """
    Save histogram and collision data to an HDF5 file with metadata.

    Parameters:
    - filename (str): Output HDF5 file path
    - stamped_histogram (np.ndarray): Histogram data [H, W, B]
    """
    with h5py.File(filename, 'w') as h5f:
        h5f.create_dataset("stamped_histogram", data=stamped_histogram, compression="gzip")

        # Add metadata
        # h5f.attrs["range_min"] = range_min
        # h5f.attrs["range_max"] = range_max
        # h5f.attrs["image_width"] = image_width
        # h5f.attrs["image_height"] = image_height
        # h5f.attrs["bin_number"] = bin_number
