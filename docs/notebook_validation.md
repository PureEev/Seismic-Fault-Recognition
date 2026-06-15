# Notebook Validation

## Scope

The repository contains 17 notebooks. The smoke test checks that:

- every code cell before the long training loop executes without an exception;
- datasets and DataLoaders return tensors with the expected shape;
- models accept the configured `128 x 128 x 128` input;
- losses are finite;
- training notebooks complete one real
  `forward -> loss -> backward -> optimizer.step`;
- validation and visualization notebooks complete all cells.

This is a startup and integration check. It does not replace a full multi-epoch
training run or convergence analysis.

## Local Inputs

The runner requires:

1. A directory with paired Thebe crops:

   ```text
   Thebe_Clean_Crops_V2/
     seis/*.npz
     fault/*.npz
   ```

2. A FaultSeg3D validation directory:

   ```text
   validation/
     seis/*.dat
     fault/*.dat
   ```

3. A checkpoint compatible with the final `swin_tiny`.

The runner selects one paired sample from each source and builds an isolated
temporary project under `/tmp`.

## Command

```bash
python scripts/validate_notebooks.py \
  --thebe-crops /path/to/Thebe_Clean_Crops_V2 \
  --faultseg-validation /path/to/faultseg3d/validation \
  --swin-checkpoint /path/to/swinunetr_tiny_checkpoint.pth \
  --keep-workdir
```

Use repeated `--notebook` arguments to run selected files:

```bash
python scripts/validate_notebooks.py \
  --notebook 04_faultseg3d_swin_tiny_pretrain.ipynb \
  --notebook 14_swinunetr_tiny.ipynb
```

The command writes `notebook-smoke-report.json` into the temporary work
directory. Original notebooks and configs are not modified.

## Last Verified State

On June 15, 2026:

- 17 of 17 notebooks passed;
- 10 training notebooks completed a real optimizer step;
- notebooks 02, 03 and 11 additionally completed a one-epoch local run;
- the Python test suite passed 35 of 35 tests.

The verification used CPU execution and one local sample per dataset.
