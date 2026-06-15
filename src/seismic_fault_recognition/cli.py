"""Command line interface for the seismic fault recognition package."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
import argparse
import json
import sys

from .logger import get_logger

logger = get_logger("sfr.cli")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``sfr`` command line interface.

    Args:
        argv: Optional argument vector. ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` on success, ``1`` for validation failures, and
        ``2`` for usage or input errors.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AttributeError:
        parser.print_help()
        return 2
    except (FileNotFoundError, ImportError, KeyError, TypeError, ValueError) as exc:
        logger.error(f"{exc}")
        return 2


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser for the package CLI."""

    parser = argparse.ArgumentParser(
        prog="sfr",
        description="Seismic Fault Recognition package CLI for recipes, config checks, audits, and validation.",
        epilog=(
            "Examples:\n"
            "  sfr recipes list\n"
            "  sfr config validate --base configs/datasphere.yaml --experiments configs/experiments\n"
            "  sfr data audit --experiment configs/experiments/01_data_cleaning_and_audit.yaml --output outputs/data_audit/audit_report.json\n"
            "  sfr checkpoint inspect checkpoints/run/best.pth --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    recipes = subparsers.add_parser(
        "recipes",
        help="Inspect registered experiment recipes",
        description="List package recipe names or show the components selected by one recipe.",
    )
    recipes_sub = recipes.add_subparsers(dest="recipes_command")
    recipes_list = recipes_sub.add_parser("list", help="List recipe names and main stage metadata")
    recipes_list.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    recipes_list.set_defaults(func=_cmd_recipes_list)
    recipes_show = recipes_sub.add_parser("show", help="Show one recipe and registry component summary")
    recipes_show.add_argument("name", metavar="NAME", help="Recipe name from `sfr recipes list`")
    recipes_show.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    recipes_show.set_defaults(func=_cmd_recipes_show)

    config = subparsers.add_parser(
        "config",
        help="Validate YAML configs",
        description="Validate the base DataSphere config and every experiment YAML in a directory.",
    )
    config_sub = config.add_subparsers(dest="config_command")
    config_validate = config_sub.add_parser("validate", help="Validate base and experiment configs")
    config_validate.add_argument("--base", default="configs/datasphere.yaml", metavar="YAML", help="Base YAML config path")
    config_validate.add_argument(
        "--experiments",
        default="configs/experiments",
        metavar="DIR",
        help="Directory containing per-experiment YAML files",
    )
    config_validate.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    config_validate.set_defaults(func=_cmd_config_validate)

    data = subparsers.add_parser(
        "data",
        help="Inspect local data artifacts",
        description="Audit NPZ split consistency and optionally save a reproducibility manifest.",
    )
    data_sub = data.add_subparsers(dest="data_command")
    data_audit = data_sub.add_parser("audit", help="Audit paired Thebe NPZ splits")
    data_audit.add_argument("--experiment", required=True, metavar="YAML", help="Experiment YAML with data split paths")
    data_audit.add_argument("--output", required=True, metavar="JSON", help="Audit report output path")
    data_audit.add_argument("--manifest-output", default="", metavar="JSON", help="Optional experiment manifest output path")
    data_audit.add_argument("--base-dir", default=".", metavar="DIR", help="Base directory for relative paths")
    data_audit.add_argument("--max-arrays", type=int, default=None, metavar="N", help="Limit audited arrays per split")
    data_audit.set_defaults(func=_cmd_data_audit)

    checkpoint = subparsers.add_parser(
        "checkpoint",
        help="Inspect checkpoints",
        description="Inspect checkpoint payload keys and SwinUNETR compatibility clues.",
    )
    checkpoint_sub = checkpoint.add_subparsers(dest="checkpoint_command")
    checkpoint_inspect = checkpoint_sub.add_parser("inspect", help="Inspect one checkpoint file")
    checkpoint_inspect.add_argument("path", metavar="PATH", help="Checkpoint file path")
    checkpoint_inspect.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    checkpoint_inspect.add_argument("--no-block-stats", action="store_true", help="Skip per-block weight summaries")
    checkpoint_inspect.set_defaults(func=_cmd_checkpoint_inspect)

    validate = subparsers.add_parser(
        "validate",
        help="Run validation pipelines",
        description="Run CPU/GPU validation for segmentation or SR+segmentation checkpoints.",
    )
    validate_sub = validate.add_subparsers(dest="validate_command")
    seg = validate_sub.add_parser("segmentation", help="Validate a segmentation checkpoint on Thebe NPZ data")
    _add_validation_common_args(seg)
    seg.add_argument("--checkpoint", required=True, metavar="PATH", help="Segmentation checkpoint path")
    seg.set_defaults(func=_cmd_validate_segmentation)
    sr_seg = validate_sub.add_parser("sr-segmentation", help="Validate SR + segmentation checkpoints")
    _add_validation_common_args(sr_seg)
    sr_seg.add_argument("--sr-checkpoint", required=True, metavar="PATH", help="SR generator checkpoint path")
    sr_seg.add_argument(
        "--segmentation-checkpoint",
        required=True,
        metavar="PATH",
        help="Segmentation checkpoint path",
    )
    sr_seg.add_argument("--sr-model", default="omniseis_sr_generator", help="SR model factory name")
    sr_seg.add_argument("--segmentation-model", default="swin_tiny", help="Segmentation model factory name")
    sr_seg.set_defaults(func=_cmd_validate_sr_segmentation)

    return parser


def _cmd_recipes_list(args: argparse.Namespace) -> int:
    from .recipes import get_recipe, list_recipes

    recipes = [get_recipe(name).as_dict() for name in list_recipes()]
    if args.json:
        _print_json(recipes)
    else:
        for recipe in recipes:
            print(f"{recipe['name']}\t{recipe['stage']}\t{recipe['dataset']}\t{recipe['model_variant']}")
    return 0


def _cmd_recipes_show(args: argparse.Namespace) -> int:
    from .recipes import describe_recipe_components

    payload = describe_recipe_components(args.name)
    if args.json:
        _print_json(payload)
    else:
        recipe = payload["recipe"]
        print(f"name: {recipe['name']}")
        print(f"stage: {recipe['stage']}")
        print(f"dataset: {recipe['dataset']}")
        print(f"model_variant: {recipe['model_variant']}")
        print(f"loss_profile: {recipe['loss_profile']}")
        print(f"augmentation_profile: {recipe['augmentation_profile']}")
        print(f"trainer_profile: {recipe['trainer_profile']}")
        print(f"config_name: {recipe['config_name']}")
    return 0


def _cmd_config_validate(args: argparse.Namespace) -> int:
    from .config import validate_config_file

    results = []
    base_path = Path(args.base)
    if not base_path.exists():
        raise FileNotFoundError(f"base config does not exist: {base_path}")
    results.append({"path": str(base_path), "issues": validate_config_file(base_path)})
    experiments_path = Path(args.experiments)
    if not experiments_path.exists():
        raise FileNotFoundError(f"experiments config directory does not exist: {experiments_path}")
    if not experiments_path.is_dir():
        raise ValueError(f"experiments path is not a directory: {experiments_path}")
    for path in sorted(experiments_path.glob("*.yaml")):
        results.append({"path": str(path), "issues": validate_config_file(path, experiment=True)})
    failed = [item for item in results if item["issues"]]
    if args.json:
        _print_json({"status": "failed" if failed else "ok", "files": results})
    else:
        if failed:
            for item in failed:
                logger.warning(f"Config issues in {item['path']}:")
                for issue in item["issues"]:
                    logger.warning(f"  - {issue}")
        else:
            logger.info(f"OK: validated {len(results)} config files")
    return 1 if failed else 0


def _cmd_data_audit(args: argparse.Namespace) -> int:
    from .data import build_npz_split_audit, require_paths, save_json_report
    from .provenance import build_data_manifest, save_experiment_manifest
    from .recipes import get_recipe, load_experiment_config

    base_dir = Path(args.base_dir).resolve()
    exp_config = load_experiment_config(args.experiment)
    data = exp_config.get("data", {})
    split_paths = _split_paths_from_data(data)
    if not split_paths:
        raise ValueError("experiment data section must include at least one paired split")
    flat_paths = {f"{split}_{kind}": path for split, pair in split_paths.items() for kind, path in zip(("seis", "fault"), pair)}
    resolved = require_paths(flat_paths, base_dir=base_dir)
    resolved_splits = {
        split: (resolved[f"{split}_seis"], resolved[f"{split}_fault"])
        for split in split_paths
    }
    report = build_npz_split_audit(resolved_splits, max_arrays=args.max_arrays)
    output_path = save_json_report(args.output, report)
    manifest_path = None
    if args.manifest_output:
        recipe = get_recipe(str(exp_config.get("recipe", "")))
        data_manifest = build_data_manifest(resolved, split_name=str(exp_config.get("dataset", "")), base_dir=base_dir)
        ctx = SimpleNamespace(repo_root=base_dir, recipe=recipe, exp_config=exp_config, base_config=None)
        manifest_path = save_experiment_manifest(ctx, data_manifest, args.manifest_output)

    logger.info(f"Audit status: {report['status']}")
    logger.info(f"Report saved to: {output_path}")
    if manifest_path is not None:
        logger.info(f"Manifest saved to: {manifest_path}")
    return 0 if report["status"] == "ok" else 1


def _cmd_checkpoint_inspect(args: argparse.Namespace) -> int:
    from .checkpoints import inspect_swinunetr_checkpoint, load_checkpoint_payload, summarize_block_weights

    path = Path(args.path)
    payload = load_checkpoint_payload(path)
    checkpoint_keys = sorted(str(key) for key in payload.keys())
    result: dict[str, Any] = {
        "path": str(path),
        "checkpoint_keys": checkpoint_keys,
        "epoch": payload.get("epoch"),
        "metrics": payload.get("metrics", {}),
        "swinunetr": inspect_swinunetr_checkpoint(payload).as_dict(),
    }
    if not args.no_block_stats:
        result["block_stats"] = [item.as_dict() for item in summarize_block_weights(payload)]
    if args.json:
        _print_json(result)
    else:
        print(f"path: {result['path']}")
        print(f"checkpoint_keys: {', '.join(checkpoint_keys)}")
        print(f"epoch: {result['epoch']}")
        print(f"total_tensors: {result['swinunetr']['total_tensors']}")
        print(f"has_patch5: {result['swinunetr']['has_patch5']}")
    return 0


def _cmd_validate_segmentation(args: argparse.Namespace) -> int:
    torch = _require_torch()
    from torch.utils.data import DataLoader

    from .data import SeisFaultDataset, require_paths
    from .losses import build_loss
    from .models.factory import build_model_by_name
    from .training import load_checkpoint
    from .trainers import validate_thebe_finetune

    exp_config, validation, training, base_dir = _load_validation_context(args)
    paths = require_paths(
        {"test_seis": exp_config["data"]["test_seis"], "test_fault": exp_config["data"]["test_fault"]},
        base_dir=base_dir,
    )
    device = _resolve_device(args.device, torch)
    dataset = SeisFaultDataset(
        paths["test_seis"],
        paths["test_fault"],
        target_shape=training.get("roi_size", (128, 128, 128)),
        random_crop=False,
        seed=int(training.get("seed", 42)),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size or 1), shuffle=False, num_workers=int(args.num_workers))
    model = build_model_by_name(
        str(exp_config["model_variant"]),
        img_size=tuple(int(item) for item in training.get("roi_size", (128, 128, 128))),
        in_channels=1,
        out_channels=1,
    ).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device, strict=bool(validation.get("checkpoint_strict", True)))
    loss_fn = build_loss(str(exp_config["loss_profile"])).to(device)
    metrics = validate_thebe_finetune(
        model,
        loader,
        loss_fn,
        device=device,
        threshold=float(training.get("threshold", 0.5)),
        thresholds=validation.get("thresholds"),
    )
    _write_json(args.output, metrics)
    print(f"metrics: {args.output}")
    dataset.close()
    return 0


def _cmd_validate_sr_segmentation(args: argparse.Namespace) -> int:
    torch = _require_torch()
    from torch.utils.data import DataLoader

    from .data import SeisFaultDataset, require_paths
    from .losses import build_loss
    from .models.factory import build_model_by_name
    from .training import load_checkpoint
    from .trainers import validate_sr_segmentation_pipeline

    exp_config, validation, training, base_dir = _load_validation_context(args)
    paths = require_paths(
        {"test_seis": exp_config["data"]["test_seis"], "test_fault": exp_config["data"]["test_fault"]},
        base_dir=base_dir,
    )
    device = _resolve_device(args.device, torch)
    dataset = SeisFaultDataset(
        paths["test_seis"],
        paths["test_fault"],
        target_shape=training.get("roi_size", (128, 128, 128)),
        random_crop=False,
        seed=int(training.get("seed", 42)),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size or 1), shuffle=False, num_workers=int(args.num_workers))
    sr_model = build_model_by_name(str(args.sr_model)).to(device)
    seg_model = build_model_by_name(
        str(args.segmentation_model),
        img_size=tuple(int(item) for item in training.get("roi_size", (128, 128, 128))),
        in_channels=1,
        out_channels=1,
    ).to(device)
    strict = bool(validation.get("checkpoint_strict", True))
    load_checkpoint(args.sr_checkpoint, model=sr_model, map_location=device, strict=strict)
    load_checkpoint(args.segmentation_checkpoint, model=seg_model, map_location=device, strict=strict)
    loss_fn = build_loss(str(exp_config["loss_profile"])).to(device)
    metrics = validate_sr_segmentation_pipeline(
        sr_model,
        seg_model,
        loader,
        loss_fn=loss_fn,
        device=device,
        threshold=float(training.get("threshold", 0.5)),
        thresholds=validation.get("thresholds"),
    )
    _write_json(args.output, metrics)
    print(f"metrics: {args.output}")
    dataset.close()
    return 0


def _add_validation_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", required=True, metavar="YAML", help="Experiment YAML with test data paths")
    parser.add_argument("--output", required=True, metavar="JSON", help="Validation metrics output path")
    parser.add_argument("--base-config", default="configs/datasphere.yaml", metavar="YAML", help="Base config path")
    parser.add_argument("--base-dir", default=".", metavar="DIR", help="Base directory for relative data/checkpoint paths")
    parser.add_argument("--device", default="auto", help="Torch device, or `auto` for CUDA when available")
    parser.add_argument("--batch-size", type=int, default=1, metavar="N", help="Validation batch size")
    parser.add_argument("--num-workers", type=int, default=0, metavar="N", help="DataLoader worker count")


def _load_validation_context(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    from .config import load_config
    from .recipes import load_experiment_config

    base = load_config(args.base_config)
    exp_config = load_experiment_config(args.experiment)
    validation = asdict(base.validation)
    if isinstance(exp_config.get("validation"), dict):
        validation.update(exp_config["validation"])
    training = dict(exp_config.get("training", {}))
    return exp_config, validation, training, Path(args.base_dir).resolve()


def _split_paths_from_data(data: Any) -> dict[str, tuple[str, str]]:
    if not isinstance(data, dict):
        return {}
    splits = {}
    for split in ("train", "val", "test"):
        seis = data.get(f"{split}_seis")
        fault = data.get(f"{split}_fault")
        if seis and fault:
            splits[split] = (str(seis), str(fault))
    return splits


def _resolve_device(value: str, torch: Any) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("validation commands require PyTorch") from exc
    return torch


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))


def _write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return target


def _json_default(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
