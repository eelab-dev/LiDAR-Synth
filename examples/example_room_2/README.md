# Room 2 simulation example

This example runs the bundled scene `room_2.usda` through the Monte Carlo simulation, pixelization, and histogram stages.

Run from the repository root:

```bash
source ./activate.sh --full
synthetic-data run --config examples/example_room_2/room_2_simulation.json
```

The pipeline writes these files into `examples/example_room_2/`:

- `raw.h5` — simulated photon events
- `photons.h5` — 60×60 pixelized timestamps
- `histogram.h5` — 80-bin sensor histogram

To inspect the commands without running them:

```bash
synthetic-data run --config examples/example_room_2/room_2_simulation.json --dry-run
```

The plotting notebooks use the generated files:

```bash
jupyter lab examples/example_room_2/depth.ipynb
jupyter lab examples/example_room_2/intensity.ipynb
```

The scene and pipeline settings are defined in `room_2.usda` and `room_2_simulation.json`. Edit those files and rerun the pipeline to regenerate the outputs.
