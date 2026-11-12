# Project 47 - Automatic Defect Counting and Spatial Density Estimation from CT Volumes

Single notebook: `Project47_End_to_End.ipynb` (synthetic CT generation -> density maps -> 3D Mini U-Net -> sliding-window inference -> Otsu+CC baseline -> evaluation/experiments).

## Install (Windows, Python 3.9+)
```
pip install -r requirements.txt
```
For a CUDA GPU, install PyTorch with the command from https://pytorch.org/get-started/locally/ first, then the rest. The CPU build works but is slow for 3D training.

## Run
1. `jupyter notebook` -> open `Project47_End_to_End.ipynb`.
2. Kernel -> **Restart & Run All** (cells are in pipeline order).
3. Default is the **6-volume pilot** (smoke test, 2 epochs). For the full dataset set `N_VOLUMES = 350` in the first code cell and run again.

## Fixed paths (defined once, first code cell)
```
DATA_ROOT = C:\Users\bpgod\Downloads\proejct-47\data
  training-data/ validation-data/ testing-data/   each: volumes/ density_maps/ pore_masks/ centroids/ metadata.csv
outputs/ (next to data/): checkpoints/ logs/ figures/ results/ predictions/
```
Existing data is never overwritten unless `OVERWRITE_EXISTING_DATA = True`. If the folders hold a dataset made with a different configuration (e.g. the pilot) the notebook stops with an explanatory error.

## Notes
- Full dataset = 245 / 52 / 53 volumes (train/val/test), stratified by regime and count bin, split before any patch extraction. About 6 GB on disk and roughly 10 minutes to generate.
- Windows safety: `num_workers=0`, float32 everywhere (asserted), no mixed precision.
- Training scale: targets x1000; every count = scaled integral / 1000; final conv bias = -6.0 with Softplus.
- Experiments: A (noise sweep) runs by default on up to `EXP_A_MAX_VOLUMES` test volumes; C (sigma sensitivity) and D (count extrapolation) retrain models, so enable `RUN_EXPERIMENT_C` / `RUN_EXPERIMENT_D` explicitly. Project 40 comparison is not implemented (no detector code supplied).
- Metrics are only valid for the full dataset; the pilot has a single test volume.
