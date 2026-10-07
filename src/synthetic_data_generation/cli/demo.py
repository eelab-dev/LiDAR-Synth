"""Generate tiny artificial inputs and exercise the complete sensor pipeline."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def create_inputs(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype([("distance", np.float32), ("collision_count", np.int32)])
    with h5py.File(output / "photons.h5", "w") as handle:
        dataset = handle.create_dataset("photon_data", (2, 2), dtype=h5py.vlen_dtype(dtype))
        for row, col in np.ndindex(2, 2):
            photons = np.empty(24, dtype=dtype)
            photons["distance"] = np.linspace(450, 550, 24) + row * 80 + col * 40
            photons["collision_count"] = 2
            photons["collision_count"][-4:] = 3
            dataset[row, col] = photons
    diffuse = np.zeros((2, 2), dtype=[("diffuse_avg", np.float32)])
    diffuse["diffuse_avg"] = [[0.2, 0.4], [0.6, 0.8]]
    with h5py.File(output / "diffuse.h5", "w") as handle:
        handle["diffuse_data"] = diffuse
    with h5py.File(output / "dnl.h5", "w") as handle:
        handle["dnl_matrix"] = np.tile(np.array([-0.02, 0.02]), (2, 2, 16))
    with h5py.File(output / "dark.h5", "w") as handle:
        handle["stamped_histogram"] = np.full((2, 2, 32), 4, dtype=np.int64)
    with h5py.File(output / "spatial.h5", "w") as handle:
        handle["spatial_ratio_map"] = np.array([[0.8, 1.0], [1.0, 1.2]])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a 2x2 synthetic sensor example; no GPU required.")
    parser.add_argument("--output-dir", default="outputs/demo")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    create_inputs(output)
    config = {
        "output_dir": ".",
        "stages": [{"command": "inject-noise", "options": {
            "input_pixelized_file": "photons.h5", "input_diffuse_file": "diffuse.h5",
            "input_dnl_file": "dnl.h5", "input_dark_histogram_file": "dark.h5",
            "input_spatial_ratio_map": "spatial.h5", "dark_count_scale": 25,
            "diffuse_noise_level": 10, "time_jitter_sigma": 80, "emission_mean": 2375,
            "relaxation_tau": 50, "seed": 42, "ambient_level": 20,
            "pulse_train_num": 100, "dead_time": 13000, "worker_threads": 2,
            "min_range": 0, "bin_width": 100, "bin_number": 32,
            "output_file": "histogram.h5", "output_histogram_file": "histogram.h5", "output_depth_image_path": "depth.png",
        }}],
    }
    path = output / "pipeline.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    from .main import run_pipeline
    run_pipeline(["--config", str(path)])
    print(f"Demo histogram: {output / 'histogram.h5'}")
