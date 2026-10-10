# Third-party notices

The migrated project sources retain the MIT license and original `Copyright (c) 2025 EElab.dev` notice in the repository LICENSE.

Full setup obtains [tinyusdz](https://github.com/lighttransport/tinyusdz) at revision `a04ee0bcbd1a930e30cc40938fcee3526a6fa8eb`, matching the original local simulator checkout. It is kept in the ignored `src/monte_carlo_simulation/simulator/external/tinyusdz` directory with its upstream license and notices. It is not relicensed or vendored into this repository's history.

Python and Rust dependencies retain their upstream licenses. Cargo.lock files capture the native dependency resolutions. Intel oneAPI, the SYCL GPU plugin, CUDA, and the system HDF5 installation are external prerequisites with their respective upstream terms.

`examples/scene.usda` is copied from `lidar-simulation/box_mesh.usda`. Golden sensor inputs and calibration maps in tests are tiny artificial fixtures, not captured datasets.
