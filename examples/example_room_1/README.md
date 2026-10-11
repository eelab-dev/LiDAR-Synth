# Room 1

This example uses [room_1.usda](room_1.usda) to generate photon data, depth and intensity
previews, and a separate material reflectivity map.

## Generate data

After completing the [full simulator setup](../../README.md#full-simulator-setup),
run these commands from the repository root:

```bash
source ./activate.sh --full
synthetic-data run --config examples/example_room_1/room_1_simulation.json
synthetic-data run --config examples/example_room_1/room_1_reflectivity.json
```

The [simulation configuration](room_1_simulation.json) runs photon tracing, pixelization,
and histogram generation. The [reflectivity configuration](room_1_reflectivity.json) runs
`diffuse-map` separately using the saved scene and camera. Add `--dry-run` to either
command to inspect it without generating data.

Outputs are saved beside this README:

| File | Contents |
| --- | --- |
| `raw.h5` | Simulated photon events |
| `photons.h5` | 60×60 pixelized timestamps |
| `histogram.h5` | 120-bin histogram; bin width `35`, minimum range `15`, emission mean `0` |
| `reflectivity.h5` | Assigned material reflectivity averaged over valid first ray hits |

## Generate images

Use the project Python environment as the notebook kernel. JupyterLab must be
installed in that environment (`python -m pip install jupyterlab`). Launch it
from the activated shell above, then run all cells in the desired notebook:

```bash
jupyter lab examples/example_room_1/depth.ipynb
jupyter lab examples/example_room_1/intensity.ipynb
jupyter lab examples/example_room_1/reflectivity.ipynb
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
