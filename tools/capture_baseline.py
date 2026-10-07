#!/usr/bin/env python3
"""Capture portable golden baselines from the original source repositories.

This script is intentionally read-only with respect to the original projects.
It writes tiny deterministic fixtures under this repository so migrated code can
be compared against the current LIDAR-agent noise behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


THIS_REPO = Path(__file__).resolve().parents[1]
ORIGINAL_NOISE_REPO = Path(os.environ.get("ORIGINAL_NOISE_REPO", "../LIDAR-agent")).resolve()
ORIGINAL_SIM_REPO = Path(os.environ.get("ORIGINAL_SIM_REPO", "../lidar-simulation")).resolve()
ORIGINAL_SENSOR_LIB = ORIGINAL_NOISE_REPO / "sensor_lib"
FIXTURE_ROOT = THIS_REPO / "tests" / "fixtures" / "baseline_noise"

BASE_SEED = 12345
DIFFUSE_NOISE_LEVEL = 12.5
TIME_JITTER_SIGMA_PS = 0.35
EMISSION_MEAN_PS = 1
RELAXATION_TAU_PS = 0.8
AMBIENT_LEVEL_PERCENT = 50.0
PULSE_TRAIN_NUM = 2
DEAD_TIME_PS = 100.0
MIN_RANGE_PS = 600.0
BIN_WIDTH_PS = 300.0
BIN_NUMBER = 10
DARK_COUNT_SCALE_PERCENT = 40.0
SPATIAL_JITTER_SIGMA = 0.05


def _ensure_original_imports() -> None:
    for path in (ORIGINAL_SENSOR_LIB, ORIGINAL_SENSOR_LIB / "sensorRustLib"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _write_pixelized_input(path: Path) -> None:
    """Write a 2x3 pixelized photon fixture as vlen compound records."""

    photon_dtype = np.dtype(
        [
            ("distance", np.float32),
            ("collision_count", np.int32),
        ]
    )
    vlen_dtype = h5py.vlen_dtype(photon_dtype)
    photon_data = np.empty((2, 3), dtype=object)
    values = [
        [(520.0, 2), (700.0, 3), (980.0, 1)],
        [(780.0, 2)],
        [],
        [(420.0, -1), (1100.0, 4)],
        [(1400.0, 2), (1800.0, 5)],
        [(600.0, 2), (602.0, 2), (604.0, 2)],
    ]
    for index, pixel_values in enumerate(values):
        row, col = divmod(index, 3)
        records = np.empty(len(pixel_values), dtype=photon_dtype)
        if pixel_values:
            records["distance"] = [item[0] for item in pixel_values]
            records["collision_count"] = [item[1] for item in pixel_values]
        photon_data[row, col] = records

    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("photon_data", data=photon_data, dtype=vlen_dtype)
        h5f.attrs["description"] = "Tiny synthetic pixelized photon input."
        h5f.attrs["distance_units"] = "millimetres"


def _write_diffuse_input(path: Path) -> None:
    diffuse_dtype = np.dtype(
        [
            ("diffuse_avg", np.float32),
            ("valid_count", np.int32),
        ]
    )
    diffuse = np.empty((2, 3), dtype=diffuse_dtype)
    diffuse["diffuse_avg"] = np.array(
        [[0.20, 0.80, 0.10], [0.40, 1.20, 0.60]], dtype=np.float32
    )
    diffuse["valid_count"] = np.array(
        [[10, 20, 5], [8, 30, 12]], dtype=np.int32
    )
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("diffuse_data", data=diffuse)


def _write_calibration_inputs(paths: dict[str, Path]) -> None:
    dnl = np.array(
        [
            [
                [0.00, 0.10, -0.05, 0.02, 0.00, -0.03, 0.04, 0.00, 0.01, -0.02],
                [0.02, 0.00, 0.00, -0.04, 0.05, 0.00, 0.00, 0.03, -0.01, 0.00],
                [-0.03, 0.04, 0.00, 0.00, 0.02, 0.01, -0.02, 0.00, 0.00, 0.03],
            ],
            [
                [0.05, -0.02, 0.00, 0.00, 0.01, 0.04, -0.03, 0.00, 0.02, 0.00],
                [0.00, 0.01, 0.03, -0.02, 0.00, 0.00, 0.04, -0.01, 0.00, 0.02],
                [-0.01, 0.00, 0.02, 0.00, -0.03, 0.05, 0.00, 0.00, 0.01, 0.00],
            ],
        ],
        dtype=np.float64,
    )
    dark_histogram = np.array(
        [
            [
                [0, 1, 0, 2, 0, 1, 0, 1, 0, 2],
                [1, 0, 1, 0, 2, 0, 1, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 2, 0, 1, 0],
            ],
            [
                [2, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                [0, 2, 1, 0, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 2, 0, 0, 1, 1, 0],
            ],
        ],
        dtype=np.int32,
    )
    spatial_ratio_map = np.array(
        [[1.0, 0.6, 1.4], [0.75, 1.25, 1.1]], dtype=np.float32
    )

    with h5py.File(paths["dnl"], "w") as h5f:
        h5f.create_dataset("dnl_matrix", data=dnl)
    with h5py.File(paths["dark"], "w") as h5f:
        h5f.create_dataset("stamped_histogram", data=dark_histogram)
    with h5py.File(paths["spatial"], "w") as h5f:
        h5f.create_dataset("spatial_ratio_map", data=spatial_ratio_map)


def _write_vlen_photon_matrix(h5f: h5py.File, name: str, matrix: np.ndarray) -> None:
    sample_dtype = None
    for entry in matrix.reshape(-1):
        entry = np.asarray(entry)
        if entry.dtype.names is not None:
            sample_dtype = entry.dtype
            break
    if sample_dtype is None:
        raise ValueError(f"Cannot infer dtype for {name}")
    output = np.empty(matrix.shape, dtype=object)
    for index in np.ndindex(matrix.shape):
        output[index] = np.asarray(matrix[index], dtype=sample_dtype)
    h5f.create_dataset(name, data=output, dtype=h5py.vlen_dtype(sample_dtype))


def _run_noise_cli(
    pixelized_path: Path,
    diffuse_path: Path,
    dnl_path: Path | None,
    dark_path: Path | None,
    spatial_path: Path | None,
    output_path: Path,
    depth_path: Path | None,
    *,
    noisy: bool,
) -> dict[str, object]:
    scratch = FIXTURE_ROOT / "original_cli_work"
    (scratch / "pythonLib").mkdir(parents=True, exist_ok=True)
    command = [
        str(ORIGINAL_NOISE_REPO / ".venv" / "bin" / "python"),
        str(ORIGINAL_SENSOR_LIB / "noiseSimulationCLI.py"),
        "--input_pixelized_file",
        str(pixelized_path),
        "--input_diffuse_file",
        str(diffuse_path),
        "--diffuse_noise_level",
        str(DIFFUSE_NOISE_LEVEL if noisy else 0.0),
        "--time_jitter_sigma",
        str(TIME_JITTER_SIGMA_PS if noisy else 0.0),
        "--relaxation_tau",
        str(RELAXATION_TAU_PS if noisy else 0.0),
        "--emmission_mean",
        str(EMISSION_MEAN_PS if noisy else 0),
        "--seeds",
        str(BASE_SEED),
        "--ambient_level",
        str(AMBIENT_LEVEL_PERCENT if noisy else 0.0),
        "--pluse_train_num",
        str(PULSE_TRAIN_NUM),
        "--dead_time",
        str(DEAD_TIME_PS if noisy else 1.0),
        "--worker_threads",
        "1",
        "--min_range",
        str(MIN_RANGE_PS),
        "--bin_width",
        str(BIN_WIDTH_PS),
        "--bin_number",
        str(BIN_NUMBER),
        "--output_file",
        str(output_path.with_suffix(".unused")),
        "--output_histogram_file",
        str(output_path),
    ]
    if depth_path is not None:
        command.extend(["--output_depth_image_path", str(depth_path)])
    if noisy and dnl_path is not None:
        command.extend(["--input_dnl_file", str(dnl_path)])
    if noisy and dark_path is not None:
        command.extend(["--input_dark_histogram_file", str(dark_path)])
        command.extend(["--dark_count_scale", str(DARK_COUNT_SCALE_PERCENT)])
    if noisy and spatial_path is not None:
        command.extend(["--input_spatial_ratio_map", str(spatial_path)])
        command.extend(["--spatial_jitter_sigma", str(SPATIAL_JITTER_SIGMA)])

    if not noisy:
        command = [
            str(ORIGINAL_NOISE_REPO / ".venv" / "bin" / "python"),
            str(ORIGINAL_SENSOR_LIB / "histogramCLI.py"),
            "--input_pixelized_file", str(pixelized_path),
            "--min_range", str(MIN_RANGE_PS), "--bin_width", str(BIN_WIDTH_PS),
            "--bin_number", str(BIN_NUMBER), "--emmission_mean", "0",
            "--output_histogram_file", str(output_path),
        ]
    env = os.environ.copy()
    pythonpath_entries = [
        str(ORIGINAL_SENSOR_LIB),
        str(ORIGINAL_SENSOR_LIB / "sensorRustLib"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join([entry for entry in pythonpath_entries if entry])
    env.setdefault("MPLBACKEND", "Agg")
    result = subprocess.run(
        command,
        cwd=scratch,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Original noise CLI failed:\n"
            + " ".join(command)
            + "\nSTDOUT:\n"
            + result.stdout
            + "\nSTDERR:\n"
            + result.stderr
        )
    return {
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def _capture_stage_outputs(paths: dict[str, Path]) -> dict[str, object]:
    _ensure_original_imports()
    from sensorPythonLib.ambientLib import create_ambient_timeStamp
    from sensorPythonLib.darkCountLib import (
        add_darkCount_spatial_ratio_map,
        add_dark_histogram,
        read_darkcout_histogram,
        read_spatial_ratio_map,
    )
    from sensorPythonLib.deadTimeLib import dead_time_simulation_rust
    from sensorPythonLib.diffuseMapLib import add_percentage_noise, read_diffuse_from_h5
    from sensorPythonLib.dnlLib import read_dnl_matrix
    from sensorPythonLib.histogramLib import form_histogram_image_rust
    from sensorPythonLib.relaxationDelayLib import create_relaxation_delay
    from sensorPythonLib.timejitterLib import create_emmision_jitter
    from sensorPythonLib.timeStampLib import (
        get_timeStamp_count_sum,
        merge_pixelized_timeStamp,
        read_pixllized_timeStamp_from_h5,
    )

    seed_sequence = np.random.SeedSequence(BASE_SEED)
    stage_seeds = [
        int(seed.generate_state(1, dtype=np.uint32)[0])
        for seed in seed_sequence.spawn(6)
    ]
    (
        diffuse_seed,
        jitter_seed,
        relaxation_seed,
        ambient_seed,
        dead_time_seed,
        dark_count_seed,
    ) = stage_seeds

    input_count = get_timeStamp_count_sum(str(paths["pixelized"]))
    timestamps = read_pixllized_timeStamp_from_h5(str(paths["pixelized"]))
    diffuse = read_diffuse_from_h5(str(paths["diffuse"]))
    diffuse_noisy = add_percentage_noise(
        percentage_noise=DIFFUSE_NOISE_LEVEL,
        diffuse_map=diffuse,
        random_seed=diffuse_seed,
    )
    jittered = create_emmision_jitter(
        timestamps,
        seeds_number=jitter_seed,
        jitter_sigma=TIME_JITTER_SIGMA_PS,
        emission_mean=EMISSION_MEAN_PS,
    )
    relaxed = create_relaxation_delay(
        jittered,
        seeds_number=relaxation_seed,
        relaxation_tau=RELAXATION_TAU_PS,
    )
    ambient = create_ambient_timeStamp(
        diffuse_map_matrix=diffuse_noisy,
        num_timestamps=int(AMBIENT_LEVEL_PERCENT / 100.0 * input_count),
        min_range=MIN_RANGE_PS,
        max_range=MIN_RANGE_PS + BIN_NUMBER * BIN_WIDTH_PS,
        random_seed=ambient_seed,
    )
    merged = merge_pixelized_timeStamp(relaxed, ambient)
    dead_time, discard_count, accept_count = dead_time_simulation_rust(
        input_pixelized_timeStamp=merged,
        pluse_train_num=PULSE_TRAIN_NUM,
        dead_time=DEAD_TIME_PS,
        max_workers=1,
        random_seed=dead_time_seed,
    )
    dnl_matrix = read_dnl_matrix(paths["dnl"])
    _, histogram, true_histogram = form_histogram_image_rust(
        pixels=dead_time,
        bin_number=BIN_NUMBER,
        range_distance=[MIN_RANGE_PS, MIN_RANGE_PS + BIN_NUMBER * BIN_WIDTH_PS],
        dnl_matrix=dnl_matrix,
        num_workers=1,
    )
    dark_histogram = read_darkcout_histogram(paths["dark"])
    dark_scaled = add_dark_histogram(
        source_histogram=histogram,
        darkcount_histogram=dark_histogram,
        percenter_value=DARK_COUNT_SCALE_PERCENT,
    )
    spatial_ratio = read_spatial_ratio_map(paths["spatial"])
    dark_spatial = add_darkCount_spatial_ratio_map(
        source_histogram=histogram,
        darkcount_histogram=dark_histogram,
        spatial_ratio_map=spatial_ratio,
        percenter_value=DARK_COUNT_SCALE_PERCENT,
        seed=dark_count_seed,
        spatial_jitter_sigma=SPATIAL_JITTER_SIGMA,
    )

    with h5py.File(paths["expected"], "w") as h5f:
        h5f.attrs["source"] = str(ORIGINAL_NOISE_REPO)
        h5f.attrs["base_seed"] = BASE_SEED
        h5f.attrs["stage_seed_order"] = json.dumps(
            [
                "diffuse",
                "jitter",
                "relaxation",
                "ambient",
                "dead_time",
                "dark_count",
            ]
        )
        h5f.attrs["stage_seeds"] = json.dumps(stage_seeds)
        h5f.create_dataset("diffuse_noisy", data=diffuse_noisy)
        _write_vlen_photon_matrix(h5f, "timestamps_after_relaxation", relaxed)
        _write_vlen_photon_matrix(h5f, "ambient_timestamps", ambient)
        _write_vlen_photon_matrix(h5f, "merged_timestamps", merged)
        _write_vlen_photon_matrix(h5f, "dead_time_timestamps", dead_time)
        h5f.create_dataset("histogram_dnl", data=histogram)
        h5f.create_dataset("true_histogram_dnl", data=true_histogram)
        h5f.create_dataset("dark_scaled_histogram", data=dark_scaled)
        h5f.create_dataset("dark_spatial_histogram", data=dark_spatial)

    return {
        "input_timestamp_count": int(input_count),
        "stage_seeds": {
            "diffuse": diffuse_seed,
            "jitter": jitter_seed,
            "relaxation": relaxation_seed,
            "ambient": ambient_seed,
            "dead_time": dead_time_seed,
            "dark_count": dark_count_seed,
        },
        "dead_time": {
            "discard_count": int(discard_count),
            "accept_count": int(accept_count),
            "accept_rate": float(accept_count / (discard_count + accept_count)),
        },
        "histogram_sum": int(np.asarray(histogram).sum()),
        "dark_scaled_sum": int(np.asarray(dark_scaled).sum()),
        "dark_spatial_sum": int(np.asarray(dark_spatial).sum()),
    }


def _write_raw_pixelization_input(path: Path) -> None:
    collision_dtype = np.dtype(
        [
            ("CollisionCount", np.int32),
            ("Distance", np.float64),
            ("CollisionLocation", np.float64, (3,)),
            ("CollisionDirection", np.float64, (3,)),
            ("Camera_x", np.int32),
            ("Camera_y", np.int32),
        ]
    )
    records = np.array(
        [
            (2, 1000.0, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 0, 0),
            (3, 1200.0, (0.0, 0.0, 0.0), (0.10, 0.05, -1.0), 0, 0),
            (0, 1500.0, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 0, 0),
            (1, 1800.0, (0.0, 0.0, 0.0), (-0.12, -0.08, -1.0), 0, 0),
        ],
        dtype=collision_dtype,
    )
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("CollisionData", data=records)
        h5f.attrs["FOV_X"] = 50.0
        h5f.attrs["FOV_Y"] = 50.0
        h5f.attrs["width"] = np.uint32(4)
        h5f.attrs["height"] = np.uint32(4)


def _run_pixelization_smoke(paths: dict[str, Path]) -> dict[str, object]:
    binary_candidates = [
        ORIGINAL_SIM_REPO / "rustLib" / "target" / "release" / "lidar_pixelation",
        ORIGINAL_SIM_REPO / "rustLib" / "release" / "lidar_pixelation",
    ]
    binary = next((item for item in binary_candidates if item.is_file()), None)
    if binary is None:
        return {"status": "skipped", "reason": "lidar_pixelation binary not found"}

    command = [
        str(binary),
        "--input_file",
        str(paths["raw_pixelization_input"]),
        "--output_file",
        str(paths["pixelization_output"]),
        "--image_width",
        "4",
        "--image_height",
        "4",
        "--fov_x",
        "50",
        "--fov_y",
        "50",
    ]
    env = os.environ.copy()
    hdf5_lib_candidates = [
        Path("/home/peizhao/miniconda3/envs/lidarSimulation/lib"),
        Path("/home/peizhao/miniconda3/pkgs/hdf5-1.12.1-h2b7332f_3/lib"),
    ]
    hdf5_lib = next((item for item in hdf5_lib_candidates if (item / "libhdf5.so.200").exists()), None)
    if hdf5_lib is not None:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(hdf5_lib), env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
    result = subprocess.run(
        command,
        cwd=ORIGINAL_SIM_REPO / "rustLib",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    status = "passed" if result.returncode == 0 else "failed"
    summary: dict[str, object] = {
        "status": status,
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
    if result.returncode == 0:
        with h5py.File(paths["pixelization_output"], "r") as h5f:
            photon_data = h5f["photon_data"][:]
            summary["output_shape"] = list(photon_data.shape)
            summary["output_count_sum"] = int(
                sum(len(photon_data[index]) for index in np.ndindex(photon_data.shape))
            )
    return summary


def main() -> int:
    global ORIGINAL_NOISE_REPO, ORIGINAL_SIM_REPO, ORIGINAL_SENSOR_LIB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-repo", type=Path, default=ORIGINAL_NOISE_REPO)
    parser.add_argument("--simulator-repo", type=Path, default=ORIGINAL_SIM_REPO)
    args = parser.parse_args()
    ORIGINAL_NOISE_REPO = args.noise_repo.resolve()
    ORIGINAL_SIM_REPO = args.simulator_repo.resolve()
    ORIGINAL_SENSOR_LIB = ORIGINAL_NOISE_REPO / "sensor_lib"
    if FIXTURE_ROOT.exists():
        shutil.rmtree(FIXTURE_ROOT)
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)

    paths = {
        "pixelized": FIXTURE_ROOT / "input_pixelized.h5",
        "diffuse": FIXTURE_ROOT / "input_diffuse.h5",
        "dnl": FIXTURE_ROOT / "input_dnl.h5",
        "dark": FIXTURE_ROOT / "input_dark_histogram.h5",
        "spatial": FIXTURE_ROOT / "input_spatial_ratio_map.h5",
        "expected": FIXTURE_ROOT / "expected_stage_outputs.h5",
        "cli_clean": FIXTURE_ROOT / "expected_cli_clean_histogram.h5",
        "cli_noisy": FIXTURE_ROOT / "expected_cli_noisy_histogram.h5",
        "cli_noisy_depth": FIXTURE_ROOT / "expected_cli_noisy_depth.png",
        "raw_pixelization_input": FIXTURE_ROOT / "input_raw_collision_data.h5",
        "pixelization_output": FIXTURE_ROOT / "expected_pixelized_from_raw.h5",
    }

    _write_pixelized_input(paths["pixelized"])
    _write_diffuse_input(paths["diffuse"])
    _write_calibration_inputs(paths)
    stage_summary = _capture_stage_outputs(paths)
    clean_cli = _run_noise_cli(
        paths["pixelized"],
        paths["diffuse"],
        None,
        None,
        None,
        paths["cli_clean"],
        None,
        noisy=False,
    )
    noisy_cli = _run_noise_cli(
        paths["pixelized"],
        paths["diffuse"],
        paths["dnl"],
        paths["dark"],
        paths["spatial"],
        paths["cli_noisy"],
        paths["cli_noisy_depth"],
        noisy=True,
    )
    _write_raw_pixelization_input(paths["raw_pixelization_input"])
    pixelization_summary = _run_pixelization_smoke(paths)

    with h5py.File(paths["cli_clean"], "r") as h5f:
        clean_sum = int(h5f["stamped_histogram"][:].sum())
    with h5py.File(paths["cli_noisy"], "r") as h5f:
        noisy_sum = int(h5f["stamped_histogram"][:].sum())

    summary = {
        "description": "Golden baseline captured from original LIDAR-agent and lidar-simulation code.",
        "original_noise_repo": str(ORIGINAL_NOISE_REPO),
        "original_sim_repo": str(ORIGINAL_SIM_REPO),
        "python": sys.executable,
        "parameters": {
            "base_seed": BASE_SEED,
            "diffuse_noise_level_percent": DIFFUSE_NOISE_LEVEL,
            "time_jitter_sigma_ps": TIME_JITTER_SIGMA_PS,
            "emission_mean_ps": EMISSION_MEAN_PS,
            "relaxation_tau_ps": RELAXATION_TAU_PS,
            "ambient_level_percent": AMBIENT_LEVEL_PERCENT,
            "pulse_train_num": PULSE_TRAIN_NUM,
            "dead_time_ps": DEAD_TIME_PS,
            "min_range_ps": MIN_RANGE_PS,
            "bin_width_ps": BIN_WIDTH_PS,
            "bin_number": BIN_NUMBER,
            "dark_count_scale_percent": DARK_COUNT_SCALE_PERCENT,
            "spatial_jitter_sigma": SPATIAL_JITTER_SIGMA,
        },
        "stage_summary": stage_summary,
        "cli_clean": {
            "histogram_sum": clean_sum,
            "stdout": clean_cli["stdout"],
            "stderr": clean_cli["stderr"],
            "command": clean_cli["command"],
        },
        "cli_noisy": {
            "histogram_sum": noisy_sum,
            "stdout": noisy_cli["stdout"],
            "stderr": noisy_cli["stderr"],
            "command": noisy_cli["command"],
        },
        "pixelization_smoke": pixelization_summary,
        "files": {name: str(path.relative_to(THIS_REPO)) for name, path in paths.items()},
    }
    (FIXTURE_ROOT / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["stage_summary"], indent=2))
    print(f"Clean CLI histogram sum: {clean_sum}")
    print(f"Noisy CLI histogram sum: {noisy_sum}")
    print(f"Pixelization smoke: {pixelization_summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
