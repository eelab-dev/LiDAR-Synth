import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

def generate_random_dnl_matrix(image_width, image_height, bin_number, dnl_value = 0.1, seed=None):
    """
    Generate a randomized 3D DNL matrix.

    Parameters
    ----------
    image_width : int
    image_height : int
    bin_number : int
    dnl_value : float
        Standard deviation of DNL (LSB fraction).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dnl_matrix : ndarray
        Shape (image_width, image_height, bin_number)
    """

    rng = np.random.default_rng(seed)

    dnl_matrix = rng.normal(
        loc=0.0,
        scale=dnl_value,
        size=(image_width, image_height, bin_number)
    )
    dnl_matrix = np.clip(dnl_matrix, -1.0, 1.0)

    return dnl_matrix


def save_dnl_matrix(dnl_matrix, file_path):
    """
    Save a DNL matrix to an HDF5 file under the ``dnl_matrix`` dataset.

    Parameters
    ----------
    dnl_matrix : array-like
        DNL matrix to save. Expected to be 3D with shape
        ``(image_width, image_height, bin_number)``.
    file_path : str or Path
        Output file path. If no suffix is provided, ``.h5`` is appended.

    Returns
    -------
    Path
        The saved file path.
    """
    dnl_matrix_array = np.asarray(dnl_matrix, dtype=float)
    if dnl_matrix_array.ndim != 3:
        raise ValueError(
            "dnl_matrix must be a 3D array with shape "
            "(image_width, image_height, bin_number)."
        )

    output_path = Path(file_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".h5")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("dnl_matrix", data=dnl_matrix_array)
    return output_path


def read_dnl_matrix(file_path):
    """
    Read a DNL matrix from an HDF5 file.

    Parameters
    ----------
    file_path : str or Path
        Path to a saved ``.h5`` file containing a ``dnl_matrix`` dataset.

    Returns
    -------
    ndarray
        Loaded DNL matrix with shape
        ``(image_width, image_height, bin_number)``.
    """
    input_path = Path(file_path)
    with h5py.File(input_path, "r") as h5f:
        if "dnl_matrix" not in h5f:
            raise KeyError(f"'dnl_matrix' dataset not found in {input_path}")
        dnl_matrix = h5f["dnl_matrix"][:]

    if dnl_matrix.ndim != 3:
        raise ValueError(
            f"DNL matrix in {input_path} must be 3D, got {dnl_matrix.ndim}D."
        )

    return dnl_matrix


def extract_pixel_dnl(
    x_index: int,
    y_index: int,
    input_dnl_matrix: np.ndarray,
) -> np.ndarray:
    """Return a 1x1xB DNL slice while preserving the outer image structure."""
    input_dnl_matrix = np.asarray(input_dnl_matrix, dtype=float)

    if input_dnl_matrix.ndim != 3:
        raise ValueError("input_dnl_matrix must be a 3D DNL matrix")

    image_height, image_width, _ = input_dnl_matrix.shape
    if not 0 <= x_index < image_width:
        raise IndexError(f"x_index {x_index} is out of bounds for width {image_width}")
    if not 0 <= y_index < image_height:
        raise IndexError(f"y_index {y_index} is out of bounds for height {image_height}")

    return input_dnl_matrix[y_index:y_index + 1, x_index:x_index + 1, :]


def _normalize_dnl_rows(dnl_rows):
    """
    Normalize input to a list of 1D numpy arrays.

    Accepts:
    - 1D array-like: one row
    - 2D array-like: each row is a DNL row
    - list/tuple of 1D array-like rows
    """
    arr = np.asarray(dnl_rows, dtype=float)
    if arr.ndim == 1:
        return [arr.reshape(-1)]
    if arr.ndim == 2:
        return [arr[i].reshape(-1) for i in range(arr.shape[0])]
    raise ValueError("dnl_rows must be a 1D row or a 2D list/array of rows.")


def displayDNL(dnl_rows):
    """
    Plot one or multiple DNL rows together.

    Parameters
    ----------
    dnl_rows : array-like
        1D DNL row, or 2D/list of DNL rows.
    """
    rows = _normalize_dnl_rows(dnl_rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, row in enumerate(rows):
        x = np.arange(row.size)
        ax.plot(x, row, linewidth=1.2, label=f"DNL row {idx}")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Bin Index")
    ax.set_ylabel("DNL (LSB)")
    ax.set_title("Differential Nonlinearity (DNL)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax


def displayINL(dnl_rows):
    """
    Plot INL curves computed from one or multiple DNL rows.

    Parameters
    ----------
    dnl_rows : array-like
        1D DNL row, or 2D/list of DNL rows.
    """
    rows = _normalize_dnl_rows(dnl_rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, row in enumerate(rows):
        inl_row = np.cumsum(row)
        x = np.arange(inl_row.size)
        ax.plot(x, inl_row, linewidth=1.2, label=f"INL row {idx}")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Bin Index")
    ax.set_ylabel("INL (LSB)")
    ax.set_title("Integral Nonlinearity (INL)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax
