# Naming and interface migration

Public Python modules use snake_case under `synthetic_data_generation`; public CLI flags use kebab-case. Existing HDF5 names and numerical arguments are retained. Native internal class/method names are preserved to keep the computational diff small.

| Original entry | New public entry |
| --- | --- |
| `dataGenJason.sh`, `diffuseMapping.sh` | `synthetic-data run --config ...` with explicit stages |
| `lidarSimulation.cpp` | `synthetic-data simulate` / `src/monte_carlo_simulation/simulator/simulate.cpp` |
| `lidarSceneDiffuseMap.cpp` | `synthetic-data diffuse-map` / `src/monte_carlo_simulation/simulator/diffuse_map.cpp` |
| `lidarGroundTruth.cpp` | `synthetic-data ground-truth` / `src/monte_carlo_simulation/simulator/ground_truth.cpp` |
| `rustLib` pixelation CLI | `synthetic-data pixelize` |
| `noiseSimulationCLI.py` | `synthetic-data inject-noise` |
| `histogramCLI.py` | `synthetic-data histogram` |
| `create_emmision_jitter` | `noise.time_jitter.create_emission_jitter` |
| `create_ambient_timeStamp` | `noise.ambient.create_ambient_timestamps` |
| `read_pixllized_timeStamp_from_h5` | `io.read_pixelized_timestamps` |
| `merge_pixelized_timeStamp` | `noise.timestamps.merge_pixelized_timestamps` |
| `read_darkcout_histogram` | `io.read_dark_count_histogram` |
| `add_darkCount_spatial_ratio_map` | `noise.dark_count.add_dark_count_spatial_ratio_map` |
| `--seeds`, `--emmission_mean`, `--pluse_train_num` | `--seed`, `--emission-mean`, `--pulse-train-num` |

Function aliases refer directly to the original implementations. Original keyword argument spellings continue to work. The renamed native extension is `_synthetic_sensor`, installed as the separate `synthetic-sensor-native` distribution during setup. The Python fallback remains available, but Rust and Python backends are not claimed to consume identical random sequences.

The legacy batch JSON schema is replaced by an explicit ordered `stages` list. This makes input/output paths and the handoff between projects visible. Existing raw/pixelized/calibration HDF5 files can be reused without conversion. Model removal in the scene commands refers to geometric objects, not trained machine-learning models.
