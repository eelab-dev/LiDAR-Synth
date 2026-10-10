import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from synthetic_data_generation.cli.main import native_command, native_executable, pipeline_commands, run_pipeline


class PipelineTests(unittest.TestCase):
    def test_native_discovery_uses_reorganized_source_tree(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "simulate": "simulator/build/LiDARSimulation",
            "diffuse-map": "simulator/build/LiDARSceneDiffuseMap",
            "ground-truth": "simulator/build/LiDARGroundtruth",
            "pixelize": "pixelization/target/release/lidar_pixelation",
        }
        with patch.dict(os.environ, {"SYNTHETIC_DATA_NATIVE_DIR": ""}), \
                patch.object(Path, "is_file", return_value=True), \
                patch("synthetic_data_generation.cli.main.os.access", return_value=True):
            for command, relative in expected.items():
                with self.subTest(command=command):
                    self.assertEqual(native_executable(command),
                                     str(root / "src/monte_carlo_simulation" / relative))

    def test_paths_resolve_against_config_not_working_directory(self):
        with tempfile.TemporaryDirectory(prefix="synthetic paths ") as directory:
            config = Path(directory) / "pipeline.json"
            config.write_text(json.dumps({"output_dir": "results", "stages": [
                {"command": "histogram", "options": {
                    "input_pixelized_file": "input.h5",
                    "output_histogram_file": "{output_dir}/histogram.h5",
                    "bin_number": 32,
                }}
            ]}))
            stages = pipeline_commands(config)
            self.assertEqual(stages[0][1], [
                "--input-pixelized-file", str(config.parent / "input.h5"),
                "--output-histogram-file", str(config.parent / "results/histogram.h5"),
                "--bin-number", "32",
            ])

    def test_pipeline_stops_on_first_failed_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "pipeline.json"
            config.write_text(json.dumps({"stages": [
                {"command": "simulate"}, {"command": "pixelize"}, {"command": "inject-noise"}
            ]}))
            with patch("synthetic_data_generation.cli.main.dispatch", side_effect=ValueError("failed")) as dispatch:
                with self.assertRaisesRegex(ValueError, "failed"):
                    run_pipeline(["--config", str(config)])
                self.assertEqual(dispatch.call_count, 1)

    def test_native_adapter_preserves_explicit_values_and_leaves_defaults_native(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scene.usda"
            source.write_text("#usda 1.0\n")
            output = Path(directory) / "raw.h5"
            with patch("synthetic_data_generation.cli.main.native_executable", return_value="LiDARSimulation"):
                command = native_command("simulate", ["--scene", str(source), "--output", str(output), "--seed", "19", "--samples-per-ray", "8", "--baseline-offset", "1", "2", "3"])
            self.assertEqual(command, ["LiDARSimulation", "--model", str(source), "--output", str(output), "--ssp", "8", "--seed", "19", "--baselineOffset", "1.0", "2.0", "3.0"])
            self.assertNotIn("--width", command)

    def test_recursive_or_unknown_stages_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "pipeline.json"
            for command in ("run", "demo", "train"):
                config.write_text(json.dumps({"stages": [{"command": command}]}))
                with self.assertRaises(ValueError):
                    pipeline_commands(config)


if __name__ == "__main__":
    unittest.main()
