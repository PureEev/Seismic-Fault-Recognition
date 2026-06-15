# Training And Validation Architecture

The project is organized around reusable package modules, YAML configs and
recipe names. Notebooks are runnable drivers; the package and CLI are the stable
interfaces for audit, validation and inspection.

## Core Contract

The package does not assume network access and does not download data. Every
workflow is configured by:

- `configs/datasphere.yaml` for shared runtime, path, validation and
  reproducibility defaults;
- `configs/experiments/*.yaml` for recipe-specific data paths, checkpoints,
  outputs and hyperparameters;
- registered profiles in `recipes.py`, `losses.py`, `augmentations.py`,
  `trainers.py` and `models/*`.

The expected tensor shape for 3D training and validation loops is
`B,C,D,H,W`. Dataset wrappers normalize notebook-friendly `D,H,W` or `B,D,H,W`
cases through `normalize_batch_dims`.

## ClearML

ClearML is optional and initialized through
`seismic_fault_recognition.clearml.init_clearml_from_context(ctx)`.

| Capability | Implementation |
| --- | --- |
| Task initialization | Recipe and experiment config are connected by `init_clearml_from_context`. |
| Metric logging | `clearml_metric_logger(..., series_prefix=...)` reports returned numeric metrics. |
| Learning-rate logging | `report_optimizer_lr` reports optimizer group learning rates. |
| Offline operation | Missing or disabled ClearML returns a no-op `ClearMLRun`. |

## Stages

| Stage | Runnable interface | Reusable code |
| --- | --- | --- |
| Environment/data check | `00_environment_and_data_check.ipynb`, `sfr config validate` | `config.py`, `recipes.py`, `notebook_utils.py` |
| Data audit | `01_data_cleaning_and_audit.ipynb`, `sfr data audit` | `data.py`, `provenance.py` |
| SimMIM pretraining | `02_*`, `03_*` notebooks | `MaskGenerator3D`, `SimMIMDatasetWrapper`, `build_simmim_masking`, `train_simmim_epoch` |
| FaultSeg3D pretraining | `04_*`, `05_*` notebooks | `FaultSeg3DMemmapDataset`, `build_faultseg3d_transforms`, `train_faultseg3d_epoch`, `evaluate_faultseg3d` |
| Thebe fine-tuning | `06_*` through `10_*` notebooks | `SeisFaultDataset`, `build_thebe_crop_pipeline`, `train_thebe_finetune_epoch`, `validate_thebe_finetune` |
| SR training | `11_sr_training_seisgan.ipynb` | `SRDynamicDataset`, `build_sr_degradation_pipeline`, `OmniSeisSRGenerator`, `PatchGANDiscriminator3D`, `train_sr_epoch`, `validate_sr` |
| Segmentation validation | `12_segmentation_validation.ipynb`, `sfr validate segmentation` | `validate_segmentation_only`, `load_checkpoint`, `build_loss` |
| SR + segmentation validation | `13_end_to_end_sr_segmentation_validation.ipynb`, `sfr validate sr-segmentation` | `validate_sr_segmentation_pipeline` |
| Model variants | `14_*`, `15_*` notebooks, `sfr recipes show` | `models/swinunetr_variants.py`, `models/faultformer.py` |
| 3D visualization | `16_3d_visualization.ipynb` | `viz.py` |

## Training Interfaces

Full training remains notebook-driven in v1. Each training notebook should:

1. call `load_notebook_context(recipe_name)`;
2. resolve paths through `notebook_utils.resolve_path` or `resolve_many`;
3. seed Python, NumPy and Torch with `seed_everything`;
4. build datasets, loaders, model, loss and optimizer from package functions;
5. run the selected trainer function;
6. save checkpoints with `save_checkpoint`;
7. save separate manifests with `save_experiment_manifest` when the run needs auditability.

This keeps the package functions testable while preserving interactive training
control for R&D experiments.

## Validation Interfaces

Validation can be run from notebooks or from the CLI:

```bash
sfr validate segmentation \
  --experiment configs/experiments/12_segmentation_validation.yaml \
  --checkpoint checkpoints/segmentation_validation/best_model.pth \
  --output outputs/segmentation_validation/metrics.json
```

The CLI builds `SeisFaultDataset`, resolves `validation.thresholds` from the
base config and experiment override, restores the checkpoint using
`training.load_checkpoint`, and writes metrics as JSON.

## Data And Augmentations

| Functionality | Current location |
| --- | --- |
| Thebe paired crop, padding, z-score, binary mask | `SeisFaultDataset`, `build_thebe_crop_pipeline` |
| FaultSeg3D MONAI affine/elastic/noise transforms | `build_faultseg3d_transforms` |
| SimMIM grid masking with `mask_ratio`/`mask_patch_size` | `MaskGenerator3D`, `build_simmim_masking` |
| SR blur, depth jitter, scale mask, trace drop, SNR noise | `build_sr_degradation_pipeline` |
| SeisGAN 2D inline/crossline patches and LR/HR pairs | `preprocessing.seisgan2d` |
| Thebe chunk assembly, bounds detection and crop saving | `preprocessing.thebe_crops` |
| SEG-Y coordinate diagnostics and regular memmap binning | `preprocessing.segy_geometry` |

## Checkpoints, Metrics And Provenance

| Capability | Current location |
| --- | --- |
| Save model/optimizer/scheduler state | `training.save_checkpoint` |
| Separate reproducibility manifest | `provenance.build_data_manifest`, `provenance.save_experiment_manifest` |
| Prefix cleanup and loose loading | `checkpoints.clean_state_dict`, `checkpoints.load_state_dict_loose` |
| Checkpoint structure diagnostics | `checkpoints.inspect_swinunetr_checkpoint`, `checkpoints.summarize_block_weights` |
| Dice/IoU/F1/AP/PR-AUC threshold sweep | `metrics.segmentation_metrics_from_logits` |
| SR PSNR/SSIM with configurable normalization | `metrics.sr_quality_metrics` |

Checkpoint files intentionally remain narrow:

- `model_state_dict`;
- optional `optimizer_state_dict`;
- optional `scheduler_state_dict`;
- `epoch`;
- flat `metrics`;
- optional explicit `extra` fields supplied by the caller.

Experiment provenance is a separate JSON artifact. A typical location is
`outputs/<experiment>/experiment_manifest.json` or
`checkpoints/<experiment>/manifest.json`.

Primary segmentation model selection should use
`validation.primary_metric: val_dice_best_threshold`. The plain `f1` and
`f1@0.5` fields remain in output metrics for comparison with older result
tables and simple binary-threshold dashboards.

## Quality Gates

Run:

```bash
python3 -m compileall src tests
python3 -m unittest discover -s tests -v
jq empty notebooks/*.ipynb
sfr config validate --base configs/datasphere.yaml --experiments configs/experiments
```

For a docs-oriented smoke check of the CLI:

```bash
PYTHONPATH=src python3 -m seismic_fault_recognition.cli --help
PYTHONPATH=src python3 -m seismic_fault_recognition.cli recipes list
```
