# Native Scene Migration

This slice migrates the scene-generation, Monte Carlo LiDAR simulation, diffuse-map, ground-truth, and raw-photon pixelization code from `/home/peizhao/lidar-simulation`.

## Source Mapping

- `pythonScripts/generateCamera.py` -> `src/synthetic_data_generation/scene/generate_camera.py`
- `pythonScripts/scenGen.py` -> `src/synthetic_data_generation/scene/generate_scene.py`
- `pythonScripts/static_scene.py` -> `src/synthetic_data_generation/scene/static_scene.py`
- `pythonScripts/removeSmallBox.py` -> `src/synthetic_data_generation/scene/remove_object.py`
- `pythonScripts/pythonLib/scenGenLib.py` -> `src/synthetic_data_generation/scene/scene_generation_lib.py`
- `syclImplementation/*.cpp` -> `native/simulator/{simulate,diffuse_map,ground_truth}.cpp`; headers -> `native/simulator/include/`
- `rustLib/src/{pixelationCLI,pixlelationLib}.rs` -> `native/pixelization/src/{main,pixelization}.rs`; Cargo manifests retained

## Python Module Entry Points

The migrated scene modules can run with `python -m`:

```bash
python -m synthetic_data_generation.scene.generate_camera --camera-location-min 0 254 771.7 --camera-location-max 0 254 771.7 --yaw 0 --pitch 0 --camera-file camera.json
python -m synthetic_data_generation.scene.generate_scene --output-file scene.usda --geometry-file scene.json
python -m synthetic_data_generation.scene.static_scene
python -m synthetic_data_generation.scene.remove_object --input-file scene.usda --output-file scene_no_short_box.usda --metadata-file scene.json --output-metadata-file scene_no_short_box.json
```

Legacy underscore flags are retained as aliases for the scene tools.

## Native Targets

The simulator CMake targets remain:

- `LiDARSimulation`
- `LiDARSceneDiffuseMap`
- `LiDARGroundtruth`

The Rust pixelization binary remains:

- `lidar_pixelation`

`setup.sh --full` passes `-DSYCL_TARGETS="${SYCL_TARGETS:-nvptx64-nvidia-cuda,spir64_x86_64}"` to CMake. The simulator build expects oneAPI `icx`/`icpx`, HDF5 development libraries, and `native/simulator/external/tinyusdz` checked out at revision `a04ee0bcbd1a930e30cc40938fcee3526a6fa8eb`.

## Preservation Notes

Internal C++ and Rust algorithms, target names, HDF5 dataset names, record shapes, RNG formulas, and camera/pixelization math are preserved. The migration only changes package-relative Python imports, Python module entry points, CLI aliases, and CMake portability for `SYCL_TARGETS`.
