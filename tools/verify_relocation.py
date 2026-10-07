#!/usr/bin/env python3
"""Smoke-test copied source in a directory with spaces using this Python environment."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="synthetic relocation ") as directory:
        relocated = Path(directory)
        shutil.copytree(root / "src", relocated / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        env = dict(os.environ, PYTHONPATH=str(relocated / "src"), MPLBACKEND="Agg")
        check = "import synthetic_data_generation as p; from pathlib import Path; assert Path(p.__file__).is_relative_to(Path.cwd()); print(p.__file__)"
        subprocess.run([sys.executable, "-c", check], cwd=relocated, env=env, check=True)
        subprocess.run([sys.executable, "-m", "synthetic_data_generation", "demo", "--output-dir", "results"], cwd=relocated, env=env, check=True)
        with h5py.File(relocated / "results/histogram.h5") as handle:
            histogram = handle["stamped_histogram"][:]
            assert histogram.shape == (2, 2, 32)
            assert np.sum(histogram) > 0
        executable = root / "native/pixelization/target/release/lidar_pixelation"
        if executable.is_file():
            target = relocated / "native/pixelization/target/release/lidar_pixelation"
            target.parent.mkdir(parents=True)
            shutil.copy2(executable, target)
            raw = relocated / "raw.h5"
            shutil.copy2(root / "tests/fixtures/baseline_noise/input_raw_collision_data.h5", raw)
            subprocess.run([sys.executable, "-m", "synthetic_data_generation", "pixelize",
                            "--input", str(raw), "--output", "pixelized.h5", "--width", "4", "--height", "4",
                            "--fov-x", "50", "--fov-y", "50"], cwd=relocated, env=env, check=True)
    print("Relocated sensor and available pixelization smoke checks passed.")


if __name__ == "__main__":
    main()
