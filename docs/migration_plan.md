# Approved migration plan

Create an independent repository from the current working source snapshots of lidar-simulation and the sensor pipeline in LIDAR-agent. Preserve algorithms, defaults, units, RNG streams, array dtypes, HDF5 contracts, native backends, and stage ordering. Rename and relocate interfaces only after baseline tests. Do not modify original projects.

## Scope
Include scene/camera generation, SYCL simulation/diffuse/ground truth, Rust pixelization, sensor noise Python and Rust code, histogram/depth IO, tiny synthetic examples, tests, setup and documentation. Exclude model training/classifiers/weights, experiment logs, captured datasets, notebooks, caches and build artifacts. Source manifest records inspected source candidates; provenance will record the final selected mapping.

## Sequence and acceptance
1. Record source hashes and establish original-output fixtures before migration changes.
2. Copy and normalize modules and public interfaces, retaining on-disk schemas.
3. Connect stages through portable CLI/config paths and optional legacy aliases.
4. Provide setup.sh --noise-only and --full with explicit prerequisites and pinned dependency revisions.
5. Compare original/new outputs, run existing tests, native builds, CLI examples and relocation checks. Record environmental gaps honestly.
6. Independently review changes, document findings, and initialize/commit the local Git repository.

## Logic findings
As explicitly requested, record discovered logic errors in future_improvment.md; do not repair numerical/runtime algorithms as part of migration. Integration defects introduced by migration must be fixed.

## Data policy
Only tiny generated fixtures and minimal distributable example scenes are committed. Measured calibration and production datasets remain external configurable inputs. Preserve source MIT and third-party notices.
