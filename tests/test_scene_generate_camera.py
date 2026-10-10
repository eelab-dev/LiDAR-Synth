from __future__ import annotations

import random
import unittest

from monte_carlo_simulation.scene.generate_camera import (
    DEFAULT_BASELINE_OFFSET,
    look_at_from_yaw_pitch,
    resolve_camera_pose,
    sample_baseline_offset,
    sample_camera_position,
)


class SampleCameraPositionTests(unittest.TestCase):
    def test_samples_each_axis_within_configured_bounds(self) -> None:
        rng = random.Random(1234)

        samples = [
            sample_camera_position([-80, 174, 671.7], [80, 334, 871.7], rng)
            for _ in range(500)
        ]

        for x, y, z in samples:
            self.assertGreaterEqual(x, -80)
            self.assertLessEqual(x, 80)
            self.assertGreaterEqual(y, 174)
            self.assertLessEqual(y, 334)
            self.assertGreaterEqual(z, 671.7)
            self.assertLessEqual(z, 871.7)

    def test_equal_bounds_produce_a_fixed_position(self) -> None:
        position = sample_camera_position([1, 2, 3], [1, 2, 3], random.Random(7))

        self.assertEqual(position, [1.0, 2.0, 3.0])

    def test_rejects_a_reversed_axis_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum exceeds maximum on x"):
            sample_camera_position([2, 0, 0], [1, 1, 1])

    def test_zero_yaw_and_pitch_point_along_negative_z(self) -> None:
        look_at = look_at_from_yaw_pitch([10, 20, 30], yaw=0, pitch=0)

        self.assertAlmostEqual(look_at[0], 10)
        self.assertAlmostEqual(look_at[1], 20)
        self.assertAlmostEqual(look_at[2], 29)

    def test_positive_yaw_turns_toward_positive_x(self) -> None:
        look_at = look_at_from_yaw_pitch([10, 20, 30], yaw=90, pitch=0)

        self.assertAlmostEqual(look_at[0], 11)
        self.assertAlmostEqual(look_at[1], 20)
        self.assertAlmostEqual(look_at[2], 30)

    def test_positive_pitch_turns_upward(self) -> None:
        look_at = look_at_from_yaw_pitch([10, 20, 30], yaw=0, pitch=30)

        self.assertAlmostEqual(look_at[0], 10)
        self.assertAlmostEqual(look_at[1], 20.5)
        self.assertAlmostEqual(look_at[2], 30 - 3**0.5 / 2)

    def test_pose_derives_look_at_from_sampled_position(self) -> None:
        location, look_at = resolve_camera_pose(
            camera_location_min=[0, 254, 771.7],
            camera_location_max=[0, 254, 771.7],
            yaw=0,
            pitch=0,
            rng=random.Random(99),
        )

        self.assertEqual(location, [0.0, 254.0, 771.7])
        self.assertEqual(look_at, [0.0, 254.0, 770.7])


class SampleBaselineOffsetTests(unittest.TestCase):
    def test_missing_bounds_use_the_legacy_offset(self) -> None:
        self.assertEqual(
            sample_baseline_offset(None, None, random.Random(1)),
            DEFAULT_BASELINE_OFFSET,
        )

    def test_samples_each_axis_within_configured_bounds(self) -> None:
        samples = [
            sample_baseline_offset(
                [4.5, -0.2, -0.1],
                [5.5, 0.2, 0.1],
                random.Random(seed),
            )
            for seed in range(100)
        ]

        for x, y, z in samples:
            self.assertGreaterEqual(x, 4.5)
            self.assertLessEqual(x, 5.5)
            self.assertGreaterEqual(y, -0.2)
            self.assertLessEqual(y, 0.2)
            self.assertGreaterEqual(z, -0.1)
            self.assertLessEqual(z, 0.1)

    def test_equal_bounds_produce_a_fixed_offset(self) -> None:
        offset = sample_baseline_offset(
            [5, 0, 0], [5, 0, 0], random.Random(7)
        )

        self.assertEqual(offset, [5.0, 0.0, 0.0])

    def test_rejects_only_one_bound(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            sample_baseline_offset([5, 0, 0], None)

    def test_rejects_a_reversed_axis_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum exceeds maximum on y"):
            sample_baseline_offset([5, 1, 0], [5, 0, 0])


if __name__ == "__main__":
    unittest.main()
