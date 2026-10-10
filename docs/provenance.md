# Source provenance

This repository is a source extraction from the current working files of the two original projects. Their Git histories were not merged. Uncommitted source changes present at migration time were included, rather than reverting to HEAD. Original project source files were not modified.

- `lidar-simulation` source HEAD: `03222c9264145fad212eb0bda4949c0818d1c077`; working tree had local changes.
- `LIDAR-agent` source HEAD: `90cc9058799f813012819385b0dcb6475990327a`; working tree had local changes.

`source_manifest.json` records source-file SHA-256 hashes and the original status snapshot. It lists inspected candidates, not all of them were included. `source_verification.json` records the final unchanged-source check and structural comparisons of the migrated implementations.

## Included source mapping

| Original | New location |
| --- | --- |
| lidar-simulation `syclImplementation/lidarSimulation.cpp` | `src/monte_carlo_simulation/simulator/simulate.cpp` |
| lidar-simulation `syclImplementation/lidarSceneDiffuseMap.cpp` | `src/monte_carlo_simulation/simulator/diffuse_map.cpp` |
| lidar-simulation `syclImplementation/lidarGroundTruth.cpp` | `src/monte_carlo_simulation/simulator/ground_truth.cpp` |
| lidar-simulation `syclImplementation/include/*.hpp` | `src/monte_carlo_simulation/simulator/include/` |
| lidar-simulation `rustLib/src/` | `src/monte_carlo_simulation/pixelization/src/` (`main.rs`, `pixelization.rs`) |
| lidar-simulation camera/scene/static-scene/object-removal scripts and scenGenLib | `src/monte_carlo_simulation/scene/` |
| LIDAR-agent `sensorPythonLib` (except for_papaer.py) | `src/spad_noise_injection/` |
| LIDAR-agent noiseSimulationCLI.py / histogramCLI.py | `src/synthetic_data_generation/cli/noise.py` / `clean.py` |
| LIDAR-agent sensorRustLib sources | `src/spad_noise_injection/rust/` |
| Both projects' relevant tests | `tests/` |

The source shell orchestrators were replaced by the portable staged JSON runner and setup/activation scripts. Their machine-specific paths and file-moving deployment behavior are not carried into the new interface. Numerical processing still calls the migrated implementations. CLI aliases accept original noise flag spellings; the old top-level batch JSON schema itself is not silently reinterpreted.

## Excluded

Training code, model definitions, weights, evaluations, experiment logs, production datasets, measured calibration captures, notebooks, original environments, compiled outputs, and Git history. tinyusdz is obtained separately at its recorded revision and keeps its upstream notices.

## Golden fixtures

- `baseline_noise`: small artificial HDF5 inputs and outputs from the original sensor Rust backend/CLI and Rust pixelizer. NumPy 2.4.2, h5py 3.15.1, matplotlib 3.10.8, Python 3.12.3. Seed and stage parameters are recorded in its JSON manifest.
- `baseline_scene`: original scene generator with Python and NumPy seeds 2026, short-box rotation 17.5 degrees, Python 3.12.0, NumPy 1.26.4, OpenUSD 25.5. The migrated shared environment reproduces its scene text and metadata exactly.
- `baseline_gpu`: a small generated scene and original native outputs using camera [0,330,810], look-at [0,274,0], up [0,1,0], baseline [5,0,0], FOV 40x40, detector 20x20, and grid 4x4. Tracing uses seed 7 / 2 samples per launch; diffuse mapping uses seed 4 / 1 sample. Photon records compare as a sorted multiset because GPU append order can vary.

These fixture files are synthetic regression references and contain no captured production data. `tools/capture_baseline.py` is an optional maintainer tool requiring the original sources; normal installation, examples, and tests use only this repository.
