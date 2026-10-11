# NLOS experiment settings

This example uses [nlos_experiment_setting.usda](nlos_experiment_setting.usda) to generate photon data, depth and intensity
previews, and a separate material reflectivity map.

## Generate data

After completing the [full simulator setup](../../README.md#full-simulator-setup),
run these commands from the repository root:

```bash
source ./activate.sh --full
synthetic-data run --config examples/NLOS_experiment_setting_example/NLOS_setting_simulation.json
synthetic-data run --config examples/NLOS_experiment_setting_example/NLOS_setting_reflectivity.json
```

The [simulation configuration](NLOS_setting_simulation.json) runs photon tracing, pixelization,
and histogram generation. The [reflectivity configuration](NLOS_setting_reflectivity.json) runs
`diffuse-map` separately using the saved scene and camera. Add `--dry-run` to either
command to inspect it without generating data.

Outputs are saved beside this README:

| File | Contents |
| --- | --- |
| `raw.h5` | Simulated photon events |
| `photons.h5` | 32×32 pixelized timestamps |
| `histogram.h5` | 64-bin histogram; bin width `100`, minimum range `0`, emission mean `2300` |
| `reflectivity.h5` | Assigned material reflectivity averaged over valid first ray hits |

## Generate images

Use the project Python environment as the notebook kernel. JupyterLab must be
installed in that environment (`python -m pip install jupyterlab`). Launch it
from the activated shell above, then run all cells in the desired notebook:

```bash
jupyter lab examples/NLOS_experiment_setting_example/depth.ipynb
jupyter lab examples/NLOS_experiment_setting_example/intensity.ipynb
jupyter lab examples/NLOS_experiment_setting_example/reflectivity.ipynb
```

| Notebook | Input | Saved image |
| --- | --- | --- |
| [depth.ipynb](depth.ipynb) | `histogram.h5` | `depth.png` — strongest histogram bin, not calibrated distance |
| [intensity.ipynb](intensity.ipynb) | `histogram.h5` | `intensity.png` — photons within the histogram window |
| [reflectivity.ipynb](reflectivity.ipynb) | `reflectivity.h5` | `reflectivity.png` — material reflectivity on a fixed 0–1 scale |

The notebooks reuse existing HDF5 data by default and generate missing data.
Set `regenerate = True` after changing the scene, materials, camera, or pipeline
settings. Depth and intensity share one histogram, so regenerate it only once.
Reflectivity uses its own pipeline and must be regenerated separately. Plotting
existing data does not require the native runtime; generation does.

Generated `.h5` and `.png` files stay local and are ignored by Git. Notebook PNGs
do not automatically replace the gallery snapshots in `docs/images/`; see the
[main gallery](../../README.md#example-images).

## Object reflectivity

Each object has its own Lambertian material. These reflectivities are illustrative
simulation settings, not measured material properties.

| Object | Reflectivity |
| --- | --- |
| `floor` | 0.60 |
| `ceiling` | 0.85 |
| `back_wall` | 0.75 |
| `left_wall` | 0.35 |
| `right_wall` | 0.50 |
| `ground` | 0.25 |
| `wall` | 0.90 |
| `tall_box` | 0.15 |

The gallery reflectivity snapshot uses these materials. Its depth and intensity
snapshots predate the material change; rerun the simulation and plotting notebooks
to obtain results for the current scene.
