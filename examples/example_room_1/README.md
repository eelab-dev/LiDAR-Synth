# Room 1 simulation example

This example runs the bundled reflectivity scene `room_1.usda` through the Monte Carlo simulation, pixelization, and histogram stages.

Run from the repository root:

```bash
source ./activate.sh --full
synthetic-data run --config examples/example_room_1/room_1_simulation.json
```

The pipeline writes these files into `examples/example_room_1/`:

- `raw.h5` — simulated photon events
- `photons.h5` — 60×60 pixelized timestamps
- `histogram.h5` — 120-bin sensor histogram

To inspect the commands without running them:

```bash
synthetic-data run --config examples/example_room_1/room_1_simulation.json --dry-run
```

The plotting notebooks use the generated files:

```bash
jupyter lab examples/example_room_1/depth.ipynb
jupyter lab examples/example_room_1/intensity.ipynb
jupyter lab examples/example_room_1/reflectivity.ipynb
```

The scene and pipeline settings are defined in `room_1.usda` and `room_1_simulation.json`. Edit those files and rerun the pipeline to regenerate the outputs.
