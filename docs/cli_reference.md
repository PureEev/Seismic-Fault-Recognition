# CLI Reference

The `sfr` command is the package-first interface for registry inspection,
configuration validation, local data audits, checkpoint diagnostics and
validation runs. The same entrypoint is available as:

```bash
sfr --help
PYTHONPATH=src python3 -m seismic_fault_recognition.cli --help
```

## Exit Codes

| Code | Meaning |
| ---: | --- |
| `0` | Command completed successfully. |
| `1` | A validation or audit command completed and found failing checks. |
| `2` | Usage error, missing input path, invalid config, missing dependency or invalid checkpoint payload. |

## Dependencies

| Command group | Required extras |
| --- | --- |
| `recipes`, `config` | Base package. PyYAML is recommended; a minimal parser handles simple project YAML. |
| `data audit` | Base package plus NumPy. |
| `checkpoint inspect` | PyTorch. Install `.[train]`. |
| `validate segmentation` | PyTorch and model dependencies. Install `.[train]`. |
| `validate sr-segmentation` | PyTorch and model dependencies. Install `.[train]`. |

No CLI command downloads data or checkpoints. All paths are resolved locally.

## `sfr recipes list`

Lists registered experiment recipes with stage, dataset and model variant.

```bash
sfr recipes list
sfr recipes list --json
```

Output:

- text mode: one tab-separated row per recipe;
- JSON mode: a list of recipe dictionaries.

## `sfr recipes show NAME`

Shows one recipe and selected registry components without notebook lineage.

```bash
sfr recipes show swinunetr_thebe_finetune_raw
sfr recipes show swinunetr_thebe_finetune_raw --json
```

Required args:

- `NAME`: recipe name from `sfr recipes list`.

Output:

- text mode: recipe fields and registry selections;
- JSON mode: recipe, loss profile, augmentation profile, trainer profile and model variant summaries.

## `sfr config validate`

Validates the base config and every experiment YAML in a directory.

```bash
sfr config validate \
  --base configs/datasphere.yaml \
  --experiments configs/experiments
```

Arguments:

- `--base YAML`: base config path. Default: `configs/datasphere.yaml`.
- `--experiments DIR`: directory of experiment YAML files. Default: `configs/experiments`.
- `--json`: print machine-readable status and per-file issues.

Output:

- success: `OK: validated N config files`;
- failure: grouped schema issues per file;
- exit code `1` when any config has schema issues.

## `sfr data audit`

Audits paired Thebe-style NPZ split files defined by an experiment config.

```bash
sfr data audit \
  --experiment configs/experiments/01_data_cleaning_and_audit.yaml \
  --output outputs/data_audit/audit_report.json \
  --manifest-output outputs/data_audit/experiment_manifest.json \
  --base-dir .
```

Required args:

- `--experiment YAML`: experiment config containing `data.train_seis`,
  `data.train_fault`, `data.val_*` and/or `data.test_*` pairs.
- `--output JSON`: audit report output path.

Optional args:

- `--manifest-output JSON`: save reproducibility manifest beside the audit.
- `--base-dir DIR`: base directory for relative paths. Default: current directory.
- `--max-arrays N`: limit arrays audited per split for quick checks.

Output:

- audit report JSON with key intersections, missing pairs, shape checks, NaN/zero statistics, fault sparsity and empty-mask counts;
- optional experiment manifest JSON with data file sizes, NPZ keys/shapes/dtypes, resolved config, Git SHA and package versions;
- exit code `1` when the audit report status is `failed`.

## `sfr checkpoint inspect PATH`

Inspects a PyTorch checkpoint payload and SwinUNETR compatibility clues.

```bash
sfr checkpoint inspect checkpoints/segmentation_validation/best_model.pth
sfr checkpoint inspect checkpoints/segmentation_validation/best_model.pth --json --no-block-stats
```

Required args:

- `PATH`: checkpoint file path.

Optional args:

- `--json`: print full JSON.
- `--no-block-stats`: skip per-block weight energy and sparsity summaries.

Output:

- checkpoint payload keys, epoch and metrics when present;
- SwinUNETR hints: patch projection, patch-5 flag, input/output channels, deepest-stage presence and adapters;
- optional per-block L2 norm, near-zero sparsity and energy fraction.

## `sfr validate segmentation`

Runs segmentation validation on Thebe-style NPZ test data.

```bash
sfr validate segmentation \
  --experiment configs/experiments/12_segmentation_validation.yaml \
  --checkpoint checkpoints/segmentation_validation/best_model.pth \
  --output outputs/segmentation_validation/metrics.json \
  --device auto \
  --batch-size 1 \
  --num-workers 0
```

Required args:

- `--experiment YAML`: experiment config with `data.test_seis` and `data.test_fault`.
- `--checkpoint PATH`: segmentation checkpoint path.
- `--output JSON`: metrics output path.

Optional args:

- `--base-config YAML`: base config path. Default: `configs/datasphere.yaml`.
- `--base-dir DIR`: base directory for relative data/checkpoint paths.
- `--device DEVICE`: `auto`, `cpu`, `cuda`, or another PyTorch device string.
- `--batch-size N`: validation batch size.
- `--num-workers N`: DataLoader worker count.

Output:

- metrics JSON with loss, Dice, IoU, precision, recall, F1, AP, PR-AUC, per-threshold metrics and best-threshold metrics.

## `sfr validate sr-segmentation`

Runs end-to-end validation by passing seismic input through an SR model before
segmentation.

```bash
sfr validate sr-segmentation \
  --experiment configs/experiments/13_end_to_end_sr_segmentation_validation.yaml \
  --sr-checkpoint checkpoints/sr_training/best_model.pth \
  --segmentation-checkpoint checkpoints/segmentation_validation/best_model.pth \
  --output outputs/sr_segmentation_validation/metrics.json
```

Required args:

- `--experiment YAML`: experiment config with `data.test_seis` and `data.test_fault`.
- `--sr-checkpoint PATH`: SR generator checkpoint path.
- `--segmentation-checkpoint PATH`: segmentation checkpoint path.
- `--output JSON`: metrics output path.

Optional args:

- `--sr-model NAME`: SR model factory name. Default: `omniseis_sr_generator`.
- `--segmentation-model NAME`: segmentation model factory name. Default: `swin_tiny`.
- common validation args: `--base-config`, `--base-dir`, `--device`, `--batch-size`, `--num-workers`.

Output:

- segmentation metrics JSON computed after SR preprocessing.
