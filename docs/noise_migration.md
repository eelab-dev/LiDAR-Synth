# Noise Migration

This repo carries the sensor noise generation path from `LIDAR-agent` without
model-training or classifier code. The migrated entry points are:

- `synthetic-data inject-noise`: equivalent to `sensor_lib/noiseSimulationCLI.py`.
- `synthetic-data histogram`: equivalent to `sensor_lib/histogramCLI.py`.

The Python package keeps behavior-compatible function names where existing data
or scripts depend on them, including the serialized field name
`emmision_delay_ps`. Newer aliases such as `create_emission_jitter` and
`--emission-mean` are additive only.

The Rust native sensor backend is built as the top-level extension
`_synthetic_sensor` from `src/spad_noise_injection/rust`. Python falls back to the pure-Python
dead-time and histogram implementations if the extension is unavailable, matching
the source project fallback behavior.

Known preserved behavior to review later:

- Dark-count histograms are added after histogram formation, so those photons do
  not pass through the dead-time filter.
- Dead-time arrival uses `2 * time_ps + emmision_delay_ps + relaxation_delay_ps`,
  while histogram binning uses `time_ps + emmision_delay_ps / 2 +
  relaxation_delay_ps / 2`.
- Legacy names contain typos (`emmision`, `pixllized`, `pluse`) because they are
  part of the original public interface and HDF5 schema.
