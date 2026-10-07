# Data contracts and preserved ordering

## Boundary between simulation and sensor processing

The native simulator writes detected photon records: accumulated path distance, collision count, detector hit position, direction in the camera frame, launch coordinates and emission-delay metadata. Rust pixelization projects return directions onto pixels. Its `photon_data` HDF5 dataset is a 2D variable-length array of compound records containing `distance` and `collision_count`.

The noise loader preserves the original conversion:

```text
time_ps = distance_mm / 1000 / 2 / c * 1e12
dead-time arrival = 2 * time_ps + emmision_delay_ps + relaxation_delay_ps
histogram coordinate = time_ps + emmision_delay_ps / 2 + relaxation_delay_ps / 2
```

The simulator accumulates the whole traced path. The loader's original docstring describes a one-way path, but the formula above is the authoritative preserved computation. `min_range` and `bin_width` operate on the histogram's timing coordinate despite their inherited names. Do not interpret them as metres. Serialized spelling `emmision_delay_ps` is retained for interoperability.

## Sensor sequence

1. Load pixelized signal photons.
2. Perturb the diffuse map by uniform percentage noise, clipping negatives.
3. Append Gaussian emission jitter and exponential relaxation delay to signal photons.
4. Generate ambient photons with uniform times and diffuse-weighted spatial allocation. Their delay fields are zero and collision count is -1.
5. Merge signal and ambient photons and simulate detector dead time.
6. Bin accepted returns, adjusting bin edges using optional DNL.
7. Add optional measured dark-count noise to the formed histogram.
8. Save `stamped_histogram` and optionally a peak-bin image.

Dark counts are added after dead-time filtering. Without a spatial ratio map the measured dark histogram is scaled directly. With a spatial map the code samples Poisson counts and applies its existing spatial perturbation. Standalone random-dark helpers are retained but are not silently substituted into this CLI path.

The main seed is split into six child streams in this order: diffuse, emission jitter, relaxation, ambient, dead time, dark counts. This ordering is preserved. Both NumPy dtypes and backend selection matter to equivalence.

## Optional inputs

| Input | HDF5 dataset / field |
| --- | --- |
| Diffuse map | `diffuse_data` compound array, field `diffuse_avg` |
| DNL | `dnl_matrix`, matching spatial grid and bin count |
| Dark reference | `stamped_histogram`, matching histogram shape |
| Spatial ratio | `spatial_ratio_map`, matching spatial grid; `.npy` also supported |
| Histogram output | `stamped_histogram` |

Example calibration files are artificially generated and are not measured sensor calibration. Production calibration inputs stay outside Git. Native/Python functions retain their original spatial-axis conventions; this migration does not transpose arrays to impose a new convention.
