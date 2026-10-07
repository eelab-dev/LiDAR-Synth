import unittest

import numpy as np

from synthetic_data_generation.noise.dark_count import (
    add_random_dark_histogram,
    sample_random_dark_histogram,
)


class RandomDarkCountTests(unittest.TestCase):
    def setUp(self):
        pixel_strength = np.array(
            [
                [8000, 1000, 500, 250],
                [125, 100, 75, 50],
                [40, 30, 20, 10],
                [8, 6, 4, 2],
            ],
            dtype=np.float64,
        )
        temporal_probability = np.array([0.8, 0.15, 0.05], dtype=np.float64)
        self.reference = pixel_strength[:, :, None] * temporal_probability

    def test_seed_controls_the_complete_dark_count_sample(self):
        first = sample_random_dark_histogram(self.reference, 100, seed=123)
        repeated = sample_random_dark_histogram(self.reference, 100, seed=123)
        changed = sample_random_dark_histogram(self.reference, 100, seed=124)

        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, changed))

    def test_sample_is_not_a_scaled_copy_of_the_reference_map(self):
        sampled = sample_random_dark_histogram(self.reference, 100, seed=123)

        self.assertFalse(np.array_equal(sampled, np.rint(self.reference)))
        self.assertNotEqual(
            np.unravel_index(sampled.sum(axis=2).argmax(), sampled.shape[:2]),
            np.unravel_index(self.reference.sum(axis=2).argmax(), self.reference.shape[:2]),
        )

    def test_temporal_distribution_is_preserved_statistically(self):
        sampled = sample_random_dark_histogram(self.reference, 1000, seed=321)
        observed_probability = sampled.sum(axis=(0, 1)) / sampled.sum()

        np.testing.assert_allclose(
            observed_probability,
            np.array([0.8, 0.15, 0.05]),
            atol=0.01,
        )

    def test_random_dark_count_is_added_to_source(self):
        source = np.full(self.reference.shape, 7, dtype=np.int64)
        combined = add_random_dark_histogram(source, self.reference, 20, seed=42)

        self.assertEqual(combined.dtype, source.dtype)
        self.assertTrue(np.all(combined >= source))
        self.assertGreater(int(combined.sum()), int(source.sum()))


if __name__ == "__main__":
    unittest.main()
