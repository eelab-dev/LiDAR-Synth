from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from synthetic_data_generation.cli import clean, noise
from synthetic_data_generation.noise.ambient import create_ambient_timeStamp
from synthetic_data_generation.noise.dark_count import (
    add_darkCount_spatial_ratio_map,
    add_dark_histogram,
    read_darkcout_histogram,
    read_spatial_ratio_map,
)
from synthetic_data_generation.noise.dead_time import dead_time_simulation_rust
from synthetic_data_generation.noise.diffuse_map import add_percentage_noise, read_diffuse_from_h5
from synthetic_data_generation.noise.dnl import read_dnl_matrix
from synthetic_data_generation.noise.histogram import form_histogram_image_rust
from synthetic_data_generation.noise.relaxation_delay import create_relaxation_delay
from synthetic_data_generation.noise.time_jitter import create_emmision_jitter
from synthetic_data_generation.noise.timestamps import (
    get_timeStamp_count_sum,
    merge_pixelized_timeStamp,
    read_pixllized_timeStamp_from_h5,
)


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "baseline_noise"
SUMMARY_PATH = FIXTURE_ROOT / "baseline_summary.json"


def _summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _params() -> dict:
    return _summary()["parameters"]


def _fixture(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / _summary()["files"][name]


def _read_h5_dataset(path: Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as h5f:
        return h5f[dataset][:]


def _assert_vlen_photon_matrix_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    for index in np.ndindex(expected.shape):
        actual_pixel = np.asarray(actual[index])
        expected_pixel = np.asarray(expected[index])
        assert actual_pixel.dtype.names == expected_pixel.dtype.names
        assert len(actual_pixel) == len(expected_pixel), index
        for field in expected_pixel.dtype.names:
            if np.issubdtype(expected_pixel[field].dtype, np.floating):
                np.testing.assert_array_equal(
                    actual_pixel[field],
                    expected_pixel[field],
                    err_msg=f"{index} field {field}",
                )
            else:
                np.testing.assert_array_equal(
                    actual_pixel[field],
                    expected_pixel[field],
                    err_msg=f"{index} field {field}",
                )


def test_migrated_noise_stages_match_original_baseline() -> None:
    params = _params()
    seeds = _summary()["stage_summary"]["stage_seeds"]
    input_count = get_timeStamp_count_sum(str(_fixture("pixelized")))
    timestamps = read_pixllized_timeStamp_from_h5(str(_fixture("pixelized")))
    diffuse = read_diffuse_from_h5(str(_fixture("diffuse")))

    diffuse_noisy = add_percentage_noise(
        percentage_noise=params["diffuse_noise_level_percent"],
        diffuse_map=diffuse,
        random_seed=seeds["diffuse"],
    )
    jittered = create_emmision_jitter(
        timestamps,
        seeds_number=seeds["jitter"],
        jitter_sigma=params["time_jitter_sigma_ps"],
        emission_mean=params["emission_mean_ps"],
    )
    relaxed = create_relaxation_delay(
        jittered,
        seeds_number=seeds["relaxation"],
        relaxation_tau=params["relaxation_tau_ps"],
    )
    ambient = create_ambient_timeStamp(
        diffuse_map_matrix=diffuse_noisy,
        num_timestamps=int(params["ambient_level_percent"] / 100.0 * input_count),
        min_range=params["min_range_ps"],
        max_range=params["min_range_ps"] + params["bin_number"] * params["bin_width_ps"],
        random_seed=seeds["ambient"],
    )
    merged = merge_pixelized_timeStamp(relaxed, ambient)
    dead_time, discard_count, accept_count = dead_time_simulation_rust(
        input_pixelized_timeStamp=merged,
        pluse_train_num=params["pulse_train_num"],
        dead_time=params["dead_time_ps"],
        max_workers=1,
        random_seed=seeds["dead_time"],
    )
    dnl_matrix = read_dnl_matrix(_fixture("dnl"))
    _, histogram, true_histogram = form_histogram_image_rust(
        pixels=dead_time,
        bin_number=params["bin_number"],
        range_distance=[
            params["min_range_ps"],
            params["min_range_ps"] + params["bin_number"] * params["bin_width_ps"],
        ],
        dnl_matrix=dnl_matrix,
        num_workers=1,
    )
    dark_histogram = read_darkcout_histogram(_fixture("dark"))
    dark_scaled = add_dark_histogram(
        source_histogram=histogram,
        darkcount_histogram=dark_histogram,
        percenter_value=params["dark_count_scale_percent"],
    )
    dark_spatial = add_darkCount_spatial_ratio_map(
        source_histogram=histogram,
        darkcount_histogram=dark_histogram,
        spatial_ratio_map=read_spatial_ratio_map(_fixture("spatial")),
        percenter_value=params["dark_count_scale_percent"],
        seed=seeds["dark_count"],
        spatial_jitter_sigma=params["spatial_jitter_sigma"],
    )

    with h5py.File(_fixture("expected"), "r") as h5f:
        np.testing.assert_array_equal(diffuse_noisy, h5f["diffuse_noisy"][:])
        _assert_vlen_photon_matrix_equal(relaxed, h5f["timestamps_after_relaxation"][:])
        _assert_vlen_photon_matrix_equal(ambient, h5f["ambient_timestamps"][:])
        _assert_vlen_photon_matrix_equal(merged, h5f["merged_timestamps"][:])
        _assert_vlen_photon_matrix_equal(dead_time, h5f["dead_time_timestamps"][:])
        np.testing.assert_array_equal(histogram, h5f["histogram_dnl"][:])
        np.testing.assert_array_equal(true_histogram, h5f["true_histogram_dnl"][:])
        np.testing.assert_array_equal(dark_scaled, h5f["dark_scaled_histogram"][:])
        np.testing.assert_array_equal(dark_spatial, h5f["dark_spatial_histogram"][:])

    expected_dead_time = _summary()["stage_summary"]["dead_time"]
    assert discard_count == expected_dead_time["discard_count"]
    assert accept_count == expected_dead_time["accept_count"]


def test_migrated_noise_cli_matches_original_noisy_cli(tmp_path: Path) -> None:
    params = _params()
    output = tmp_path / "noisy_histogram.h5"
    depth = tmp_path / "noisy_depth.png"
    result = noise.main(
        [
            "--input_pixelized_file",
            str(_fixture("pixelized")),
            "--input_diffuse_file",
            str(_fixture("diffuse")),
            "--input_dnl_file",
            str(_fixture("dnl")),
            "--input_dark_histogram_file",
            str(_fixture("dark")),
            "--input_spatial_ratio_map",
            str(_fixture("spatial")),
            "--dark_count_scale",
            str(params["dark_count_scale_percent"]),
            "--spatial_jitter_sigma",
            str(params["spatial_jitter_sigma"]),
            "--diffuse_noise_level",
            str(params["diffuse_noise_level_percent"]),
            "--time_jitter_sigma",
            str(params["time_jitter_sigma_ps"]),
            "--relaxation_tau",
            str(params["relaxation_tau_ps"]),
            "--emmission_mean",
            str(params["emission_mean_ps"]),
            "--seeds",
            str(params["base_seed"]),
            "--ambient_level",
            str(params["ambient_level_percent"]),
            "--pluse_train_num",
            str(params["pulse_train_num"]),
            "--dead_time",
            str(params["dead_time_ps"]),
            "--worker_threads",
            "1",
            "--min_range",
            str(params["min_range_ps"]),
            "--bin_width",
            str(params["bin_width_ps"]),
            "--bin_number",
            str(params["bin_number"]),
            "--output_file",
            str(tmp_path / "unused.h5"),
            "--output_histogram_file",
            str(output),
            "--output_depth_image_path",
            str(depth),
        ]
    )
    assert result == 0
    np.testing.assert_array_equal(
        _read_h5_dataset(output, "stamped_histogram"),
        _read_h5_dataset(_fixture("cli_noisy"), "stamped_histogram"),
    )
    assert depth.is_file()
    from matplotlib.image import imread
    np.testing.assert_array_equal(imread(depth), imread(_fixture("cli_noisy_depth")))


def test_migrated_clean_cli_matches_original_clean_baseline(tmp_path: Path) -> None:
    params = _params()
    output = tmp_path / "clean_histogram.h5"
    result = clean.main(
        [
            "--input_pixelized_file",
            str(_fixture("pixelized")),
            "--min_range",
            str(params["min_range_ps"]),
            "--bin_width",
            str(params["bin_width_ps"]),
            "--bin_number",
            str(params["bin_number"]),
            "--emmission_mean",
            "0",
            "--output_histogram_file",
            str(output),
        ]
    )
    assert result == 0
    np.testing.assert_array_equal(
        _read_h5_dataset(output, "stamped_histogram"),
        _read_h5_dataset(_fixture("cli_clean"), "stamped_histogram"),
    )


def test_original_pixelization_smoke_fixture_was_captured() -> None:
    summary = _summary()["pixelization_smoke"]
    assert summary["status"] == "passed"
    assert summary["output_shape"] == [4, 4]
    assert summary["output_count_sum"] == 3
    output = _fixture("pixelization_output")
    with h5py.File(output, "r") as h5f:
        photon_data = h5f["photon_data"][:]
        assert photon_data.shape == (4, 4)
        assert sum(len(photon_data[index]) for index in np.ndindex(photon_data.shape)) == 3


# Use the standard-library test runner; fixture capture needs no pytest dependency.
import tempfile
import unittest
import subprocess


class BaselineRegressionTests(unittest.TestCase):
    def test_native_backend_is_available(self):
        import _synthetic_sensor
        self.assertTrue(callable(_synthetic_sensor.form_histogram_image))

    def test_noise_stages(self):
        test_migrated_noise_stages_match_original_baseline()

    def test_noisy_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            test_migrated_noise_cli_matches_original_noisy_cli(Path(directory))

    def test_clean_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            test_migrated_clean_cli_matches_original_clean_baseline(Path(directory))

    def test_pixelization_matches_original(self):
        from synthetic_data_generation.cli.main import native_executable
        try:
            executable = native_executable('pixelize')
        except FileNotFoundError:
            self.skipTest('Full setup required for native pixelization')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'pixelized.h5'
            subprocess.run([executable, '--input_file', str(_fixture('raw_pixelization_input')),
                            '--output_file', str(output), '--image_width', '4', '--image_height', '4',
                            '--fov_x', '50', '--fov_y', '50'], check=True, capture_output=True)
            with h5py.File(output) as actual, h5py.File(_fixture('pixelization_output')) as expected:
                self.assertEqual(set(actual), set(expected))
                for name in actual:
                    left, right = actual[name][:], expected[name][:]
                    if left.dtype == object:
                        _assert_vlen_photon_matrix_equal(left, right)
                    else:
                        np.testing.assert_array_equal(left, right)
