# Migration validation

Validated locally on 2026-10-07. Original source files remained unchanged, verified against the recorded SHA-256 manifest.

## Results

| Check | Result |
| --- | --- |
| `setup.sh --noise-only` with Python 3.12 | Passed; installed package and rebuilt Rust sensor extension |
| `setup.sh --full` with existing oneAPI, CUDA and custom HDF5 prefix | Passed; all three SYCL executables and Rust pixelization built |
| `RUN_GPU_TESTS=1 python -m unittest discover -s tests -v` after full activation | **30 tests passed**, no skips |
| Fixed-input noise stages | Exact array matches against original outputs, including emission/relaxation delays, diffuse noise, ambient photons, dead time, DNL histogram and both dark-count paths |
| Clean/noisy CLI and depth PNG | Exact histogram-array and rendered-image-array matches |
| Rust pixelization | Output datasets match original executable on a small raw photon fixture |
| Seeded scene generation | Scene text and geometry metadata match original OpenUSD environment exactly |
| GPU photon tracing | 74 complete photon records match original executable as a sorted multiset; repeated migrated run also matches |
| GPU diffuse mapping | 4x4 structured output matches original and repeated migrated run exactly |
| Complete example pipeline | Scene → GPU tracing → pixelization → GPU diffuse map → noise → histogram/depth completed |
| Relocation | Copied package executes sensor demo and native pixelization from a temporary directory containing spaces |
| Static/package checks | Python compileall, Bash syntax checks, pip dependency check and staged Git whitespace check passed |

Commands to reproduce:

```bash
source ./activate.sh --full
RUN_GPU_TESTS=1 python -m unittest discover -s tests -v
python tools/verify_relocation.py
synthetic-data run --config configs/pipeline.json
python -m compileall -q src tests tools
bash -n setup.sh activate.sh
python -m pip check
```

Noise-only users can run the same unittest discovery without `RUN_GPU_TESTS=1`; GPU tests are skipped, as are optional scene/pixelization tests if their full-setup prerequisites are absent.

## Environment and bounds

Reference environment: Python 3.12.3, NumPy 2.4.2, h5py 3.15.1, matplotlib 3.10.8, OpenUSD 25.5, oneAPI compiler 2024.1, CUDA-target SYCL, NVIDIA RTX 4090, and the oneAPI CPU OpenCL runtime on an AMD Threadripper PRO 5995WX. HDF5 was supplied by an existing local installation using `HDF5_DIR`; no system packages or drivers were changed.

The initial `nvidia-smi` check reported a driver/NVML version mismatch. Actual SYCL GPU execution succeeded. A separate initial runtime attempt failed at the simulator's CPU queue because only the GPU runtime was visible. Loading the oneAPI environment through `activate.sh --full` exposed both devices and the full run passed. These were environment observations, not modifications to the tracing or device-selection algorithms.

The native compiler reports inherited warnings, including an uninitialized material-binding index; they are recorded in `future_improvment.md` rather than repaired. Native-source comparison confirms 29 source files match the originals after whitespace and module/file-name normalization. Of the sensor functions, 55 ASTs are identical; the remaining differences are one docstring whitespace change and two backend import-path migrations (details in `source_verification.json`).

These checks establish equivalence for the captured fixtures and local reference environment. They do not promise bitwise reproducibility across different GPUs, compilers or arbitrary environments. GPU record order is intentionally not normalized in production, and independently sampled scenes can differ. The relocation check reuses the installed dependency environment while importing all project Python code from its copied location.
