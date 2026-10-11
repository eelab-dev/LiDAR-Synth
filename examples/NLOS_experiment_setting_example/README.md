# TMF static scene example

The bundled `nlos_experiment_setting.usda` is the positive static-scene model. 
Settings preserve its saved camera, 52.8° field of view, 55-unit baseline, 
23,000 samples per ray, 32×32 pixelization, and a 64-bin histogram with 60-unit

Run from the repository root after full setup:

```bash
source ./activate.sh --full
synthetic-data run --config examples/experiment_setting_example/NLOS_setting_simulation.json
```
