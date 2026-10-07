"""Compare a seeded scene against the original OpenUSD environment's output."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("pxr"), "Full setup required for OpenUSD")
class SceneBaselineTests(unittest.TestCase):
    def test_seeded_scene_and_metadata_match_original(self):
        from synthetic_data_generation.scene.generate_scene import main
        expected = Path(__file__).parent / "fixtures" / "baseline_scene"
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            random.seed(2026)
            np.random.seed(2026)
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                main(["--output-file", str(output / "scene.usda"),
                      "--geometry-file", str(output / "geometry.json"),
                      "--short-box-rotation", "17.5"])
                self.assertEqual((output / "scene.usda").read_text(), (expected / "scene.usda").read_text())
                self.assertEqual(json.loads((output / "geometry.json").read_text()),
                                 json.loads((expected / "geometry.json").read_text()))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
