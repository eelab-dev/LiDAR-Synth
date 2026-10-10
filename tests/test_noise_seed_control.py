import unittest

import numpy as np

from spad_noise_injection.ambient import create_ambient_timeStamp
from spad_noise_injection.diffuse_map import add_percentage_noise


class NoiseSeedControlTests(unittest.TestCase):
    def test_diffuse_noise_is_reproducible_and_seed_sensitive(self):
        diffuse = np.arange(1, 17, dtype=np.float32).reshape(4, 4)

        first = add_percentage_noise(10, diffuse, random_seed=123)
        repeated = add_percentage_noise(10, diffuse, random_seed=123)
        changed = add_percentage_noise(10, diffuse, random_seed=124)

        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, changed))

    def test_ambient_timestamps_are_reproducible_and_seed_sensitive(self):
        diffuse = np.ones((2, 2), dtype=np.float32)

        first = create_ambient_timeStamp(diffuse, 0, 6400, 40, random_seed=123)
        repeated = create_ambient_timeStamp(diffuse, 0, 6400, 40, random_seed=123)
        changed = create_ambient_timeStamp(diffuse, 0, 6400, 40, random_seed=124)

        for index in np.ndindex(first.shape):
            np.testing.assert_array_equal(first[index], repeated[index])
        self.assertTrue(
            any(
                not np.array_equal(first[index], changed[index])
                for index in np.ndindex(first.shape)
            )
        )


if __name__ == "__main__":
    unittest.main()
