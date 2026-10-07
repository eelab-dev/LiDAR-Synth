"""Optional hardware tests against the original simulator's small captured outputs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import h5py
import numpy as np

from synthetic_data_generation.cli.main import native_command


@unittest.skipUnless(os.environ.get("RUN_GPU_TESTS") == "1", "Set RUN_GPU_TESTS=1 after full activation")
class NativeEquivalenceTests(unittest.TestCase):
    def compare_stage(self, stage, filename, dataset, samples):
        fixture = Path(__file__).parent / "fixtures/baseline_gpu"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / filename
            args = ["--scene", str(fixture / "scene.usda"), "--output", str(output),
                    "--width", "4", "--height", "4", "--samples-per-ray", str(samples),
                    "--seed", "7" if stage == "simulate" else "4",
                    "--camera-position", "0", "330", "810", "--look-at", "0", "274", "0",
                    "--up", "0", "1", "0", "--baseline-offset", "5", "0", "0",
                    "--fov-x", "40", "--fov-y", "40", "--detector-width", "20", "--detector-height", "20"]
            result = subprocess.run(native_command(stage, args), capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with h5py.File(output) as actual, h5py.File(fixture / filename) as expected:
                left, right = actual[dataset][:], expected[dataset][:]
                if left.ndim == 1 and left.dtype.names:
                    # Atomic append order is not part of the photon multiset contract.
                    left = np.sort(left, order=left.dtype.names)
                    right = np.sort(right, order=right.dtype.names)
                np.testing.assert_array_equal(left, right)

    def test_photons_match_original(self):
        self.compare_stage("simulate", "raw.h5", "CollisionData", 2)

    def test_diffuse_map_matches_original(self):
        self.compare_stage("diffuse-map", "diffuse.h5", "diffuse_data", 1)
