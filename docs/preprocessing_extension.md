# Preprocessing Reference

Preprocessing utilities live under `seismic_fault_recognition.preprocessing`.
They are deterministic importable helpers for preparing local seismic datasets;
they do not download data or write outside the requested output paths.

## `preprocessing/seisgan2d.py`

Inputs:

- a NumPy 3D cube whose axes can be mapped to inline, crossline and depth;
- patch size, slice stride and window stride;
- optional degradation parameters such as Ricker frequency and SNR.

Outputs:

- `Slice2DRecord` objects with patch metadata, or plain 2D arrays;
- LR/HR pairs normalized to `[0, 1]`;
- optional `low/case_*.h5` and `high/case_*.h5` files.

Dependencies:

- NumPy for extraction and degradations;
- `h5py` only when saving HDF5 files.

Reusable functionality:

- extract inline and crossline 2D patches from a 3D cube;
- keep patch metadata through `Slice2DRecord`;
- filter empty slices by zero ratio and variance;
- min-max normalize slices;
- generate Ricker low-pass degradation along the time/depth axis;
- add SNR-controlled Gaussian noise;
- apply TSP-style spatial/time downsampling;
- create LR/HR pairs for SeisGAN-style training;
- optionally save LR/HR pairs as HDF5 files through `h5py`.

```python
from seismic_fault_recognition.preprocessing.seisgan2d import (
    extract_inline_crossline_slices,
    make_lr_hr_pairs,
)

slices = extract_inline_crossline_slices(cube, patch=(256, 256), slice_stride=16, stride=128)
pairs = make_lr_hr_pairs(slices, f0=15, snr_db=10, seed=42)
```

Artifact layout:

```text
data/seisgan_pairs/
  low/case_0.h5
  high/case_0.h5
```

## `preprocessing/thebe_crops.py`

Inputs:

- numbered `.npz` chunks or already assembled NumPy/memmap volumes;
- paired seismic and fault arrays with matching geometry;
- crop size, overlap, footprint threshold and minimum seismic standard deviation.

Outputs:

- `seis/*.npz` and `fault/*.npz` paired crop files;
- an `ExtractionReport` with saved/skipped counts and crop records.

Dependencies:

- NumPy only.

Reusable functionality:

- concatenate numbered `.npz` chunks into one `.npy` memmap;
- detect valid data bounds without loading the whole cube into RAM;
- build a footprint-aware 3D crop grid;
- save paired seismic/fault crops under `seis/` and `fault/`;
- return a structured `ExtractionReport`.

```python
from seismic_fault_recognition.preprocessing.thebe_crops import save_valid_crops

report = save_valid_crops(
    seis_volume,
    fault_volume,
    "data/thebe_clean_crops",
    crop_size=(128, 128, 128),
    overlap=0.25,
    prefix="thebe",
)
print(report.as_dict())
```

Artifact layout:

```text
data/thebe_clean_crops/
  seis/thebe_x0_y0_z0.npz
  fault/thebe_x0_y0_z0.npz
```

## `preprocessing/segy_geometry.py`

Inputs:

- SEG-Y file path for header diagnostics;
- trace memmap and coordinate `.npy` arrays for regular-grid binning.

Outputs:

- scalar candidate scores;
- `GridGeometry` estimates;
- regular cube and count memmaps from `build_regular_cube_memmap`.

Reusable functionality:

- coordinate scalar handling;
- coordinate multiplier diagnostics;
- survey orientation and bin-step estimation from CDP coordinates;
- regular-grid memmap binning from trace and coordinate arrays.

`segyio` is imported only inside SEG-Y reader functions. Importing the package
does not require SEG-Y dependencies.

Example:

```python
from seismic_fault_recognition.preprocessing.segy_geometry import (
    detect_scalar_and_coords,
    estimate_grid_geometry,
)

best_scalar, candidates = detect_scalar_and_coords("survey.sgy")
geometry = estimate_grid_geometry("survey.sgy", max_scan=100_000)
print(best_scalar, geometry.as_dict())
```

## Checkpoint Diagnostics

Checkpoint compatibility helpers are in `seismic_fault_recognition.checkpoints`:

- strip wrapper prefixes such as `module.`;
- extract state dicts from common checkpoint payloads;
- try strict loading first and fall back to non-strict loading;
- inspect SwinUNETR checkpoint features such as patch-5 projection, missing
  deepest stage, adapters and output channels;
- summarize per-block weight energy and near-zero sparsity.

Use the CLI for local inspection:

```bash
sfr checkpoint inspect checkpoints/segmentation_validation/best_model.pth --json
```

The JSON output is suitable for storing in `outputs/<experiment>/checkpoint_inspection.json`.
