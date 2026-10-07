# Future improvements

This file records existing logic issues found during migration. Their behavior is intentionally preserved. Migration integration fixes are separate.

## Photon buffer overflow can silently lose detections

- Source: `lidar-simulation/syclImplementation/lidarSimulation.cpp`, the `recordSize` allocation and atomic detector-hit counter.
- The hit counter advances even when its index exceeds the fixed record capacity. Only in-range records are stored; the final count can therefore exceed the saved photons. The output does not mark this truncation.
- Future work: detect overflow explicitly and design a chunked writer or bounded counter with a documented output policy. Preserve the existing buffer size and tracing behavior in this migration.

## Empty accepted-photon sets can cause division by zero

- Source: original `sensor_lib/noiseSimulationCLI.py`, `accept_num / (discard_num + accept_num)`.
- If both counts are zero, reporting the acceptance ratio raises an exception before the histogram is saved.
- Future work: define and test the empty-input contract. Do not silently select a new behavior in this migration.

## Small output dimensions still allocate a large simulation

- Source: native `include/Utility.hpp::computeAdjustedSize()` and `lidarSimulation.cpp`.
- The launch grid expands to approximately 500,000 positions regardless of a small requested output grid. The photon-record capacity is fixed at `640 * 120 * 800` records.
- This is an inherited resource limitation, not a migration regression. Future work: expose separately controlled launch resolution and record capacity while preserving reference defaults.

## Whole-scene reproducibility is weaker than fixed-input sensor reproducibility

- Scene generation uses random draws without a single exposed seed controlling every generator. Native detection records are appended using a GPU atomic counter, so record ordering can depend on scheduling.
- Per-photon timing noise consumes RNG draws in input order. Identical sets of native returns in a different order may lead to different noisy outputs.
- Future work: define an explicit scene seed and stable photon identity/order policy before claiming identical end-to-end results across GPU runs. This migration compares sensor outputs from identical ordered inputs and keeps the source ordering unchanged.

## Depth image values are peak-bin indices

- Source: `sensorPythonLib/depthLib.py::form_depth_from_histogram()`.
- The default estimator returns `argmax` indices without a conversion to metres, timing-offset subtraction, or a no-return sentinel for all-zero pixels (which return bin zero).
- Future work: add a separately named calibrated depth output and an explicit invalid-pixel policy. Existing image values remain unchanged.

## Noise CLI requires an unused output argument

- The original CLI validates `output_file`, but writes its histogram to `output_histogram_file` instead. The new stage interface retains both arguments and their original validation; examples supply the same filename for both.
- Future work: deprecate the unused option through a deliberate interface migration.

## Timing conversion documentation differs from the producer

- The original timestamp conversion docstring says “one-way path length,” but the native simulator stores the accumulated path to the detector and the loader divides by two before dividing by light speed.
- The actual formula and its later factors of two are preserved and documented in `docs/data_contracts.md`. Future work: clarify the coordinate's name and units throughout the original APIs without silently changing values.

## Material-binding lookup passes an uninitialized index by value

- `native/simulator/include/usd_obj_loader.hpp` declares `meshMaterialIndex` without initialization and passes it by value into `resolve_material_binding_path()`. The helper assigns only its local copy. The compiler warns about the call.
- Future work: remove the unused parameter or define its initialization and output contract after adding a focused loader test. This migration leaves the source computation unchanged.

## Simulator output also requires a CPU SYCL device

- `native/simulator/simulate.cpp` creates a CPU-selected queue at the output stage even after its GPU kernel succeeds.
- A shell exposing only the GPU aborts there. Full activation now loads the existing oneAPI GPU and CPU runtimes; no device-selection logic was changed. Future work: determine whether that CPU queue is necessary and define a runtime contract before changing it.

## Scene and native simulation

- `native/simulator/lidarSimulation.cpp` keeps the original fixed photon record capacity `640 * 120 * 800`. Larger simulations can silently stop recording hits once the atomic counter exceeds that capacity because overflow records are skipped. The migration preserves this capacity and behavior.
- `native/simulator/include/Utility.hpp` keeps `computeAdjustedSize(..., target_area = 500000)`. Small requested image sizes are expanded to roughly 500k kernel pixels before collapsing back to requested camera pixels, which can make apparently small runs expensive. This is preserved for output consistency.
- `native/simulator/lidarSimulation.cpp` still creates a GPU queue with `sycl::gpu_selector_v`. On machines without a working SYCL GPU runtime the executable will fail at runtime instead of falling back to CPU.
- `native/pixelization/src/pixlelationLib.rs` preserves the original unguarded ray direction normalization. A zero-length direction can panic during `Ray::new`; the caller catches the panic and records a failed line, preserving the source behavior.
