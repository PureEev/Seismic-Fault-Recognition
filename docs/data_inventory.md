# Data Layout

Large seismic volumes, generated crops, model checkpoints and validation
outputs are intentionally stored outside Git. Repository configs contain
repository-relative example paths that should be adapted to the local machine.

## Expected Layout

```text
data/
  thebe/
    raw/
      thebe_train_seis.npz
      thebe_train_fault.npz
      thebe_val_seis.npz
      thebe_val_fault.npz
    clean/
    clean_sr/
    sr/
    thebe_test_seis.npz
    thebe_test_fault.npz
  faultseg3d/
    train/
      seis/*.dat
      fault/*.dat
    validation/
      seis/*.dat
      fault/*.dat
checkpoints/
  <experiment>/best_model.pth
outputs/
  <experiment>/
```

Thebe NPZ files may contain one or more named 3D arrays. Seismic and fault
archives must use matching keys. FaultSeg3D directories must contain matching
seismic and label files with the configured shape and dtype.

## Path Configuration

Each `configs/experiments/*.yaml` file has three relevant sections:

- `data`: train, validation and test paths;
- `checkpoints`: input and output checkpoint paths;
- `outputs`: reports, metrics and visualization output directory.

Relative paths are resolved from the repository root. No notebook downloads
data or model weights.

## Input Contract

The main segmentation and SR workflows use:

- tensor layout `B,C,D,H,W`;
- one input channel by default;
- spatial patch size `128 x 128 x 128`;
- binary fault masks for segmentation.

The final `swin_tiny` implementation rejects other spatial input sizes.

## Data Audit

Run the audit before training:

```bash
sfr data audit \
  --experiment configs/experiments/01_data_cleaning_and_audit.yaml \
  --output outputs/data_audit/audit_report.json \
  --manifest-output outputs/data_audit/experiment_manifest.json
```

The report checks:

- missing seismic/fault keys;
- intersections between train, validation and test keys;
- shapes and dtypes;
- NaN and zero fractions;
- fault class balance and empty masks.

## Git Policy

The `.gitignore` excludes common seismic formats, NumPy archives, memmaps,
checkpoints, visualizations and experiment outputs. Public documentation should
describe where data can be obtained, but should not contain private machine
paths or credentials.
