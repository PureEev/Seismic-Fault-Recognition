#!/usr/bin/env python3
"""Execute every project notebook against isolated one-sample smoke data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import numpy as np
import yaml


TRAINING_STARTUP_CELLS = {
    "02_simmim_swinunetr_thebe_pretrain.ipynb": """
import math
train_loss = train_simmim_epoch(
    model, train_loader, optimizer, loss_fn, device=device, amp=False
)
assert math.isfinite(float(train_loss))
print(f"training_startup=ok loss={train_loss:.6f}")
""",
    "03_simmim_omniseis_thebe_pretrain.ipynb": """
import math
train_loss = train_simmim_epoch(
    model, train_loader, optimizer, loss_fn, device=device, amp=False
)
assert math.isfinite(float(train_loss))
print(f"training_startup=ok loss={train_loss:.6f}")
""",
    "04_faultseg3d_swin_tiny_pretrain.ipynb": """
import math
train_loss = train_faultseg3d_epoch(
    model, train_loader, optimizer, criterion, device=device, amp=False
)
assert math.isfinite(float(train_loss))
print(f"training_startup=ok loss={train_loss:.6f}")
""",
    "05_faultseg3d_omniseis_pretrain.ipynb": """
import math
train_loss = train_faultseg3d_epoch(
    model, train_loader, optimizer, criterion, device=device, amp=False
)
assert math.isfinite(float(train_loss))
print(f"training_startup=ok loss={train_loss:.6f}")
""",
    "06_swinunetr_thebe_finetune_raw.ipynb": """
import math
train_loss = train_thebe_finetune_epoch(
    model, train_loader, optimizer, loss_fn, device=device, amp=False, accumulation_steps=1
)
assert math.isfinite(float(train_loss))
print(f"training_startup=ok loss={train_loss:.6f}")
""",
    "07_swinunetr_thebe_finetune_clean_sr_cubes.ipynb": """
import math
train_loss = train_thebe_finetune_epoch(
    model, train_loader, optimizer, loss_fn, device=device, amp=False, accumulation_steps=1
)
assert math.isfinite(float(train_loss))
print(f"training_startup=ok loss={train_loss:.6f}")
""",
    "08_omniseis_thebe_finetune_raw.ipynb": """
import math
train_loss = train_thebe_finetune_epoch(
    model, train_loader, optimizer, loss_fn, device=device, amp=False, accumulation_steps=1
)
assert math.isfinite(float(train_loss))
print(f"training_startup=ok loss={train_loss:.6f}")
""",
    "09_omniseis_thebe_finetune_clean_cubes.ipynb": """
import math
train_loss = train_thebe_finetune_epoch(
    model, train_loader, optimizer, loss_fn, device=device, amp=False, accumulation_steps=1
)
assert math.isfinite(float(train_loss))
print(f"training_startup=ok loss={train_loss:.6f}")
""",
    "10_omniseis_thebe_finetune_clean_sr_aug_reg.ipynb": """
import math
train_loss = train_thebe_finetune_epoch(
    model, train_loader, optimizer, loss_fn, device=device, amp=False, accumulation_steps=1
)
assert math.isfinite(float(train_loss))
print(f"training_startup=ok loss={train_loss:.6f}")
""",
    "11_sr_training_seisgan.ipynb": """
import math
train_metrics = train_sr_epoch(
    generator,
    train_loader,
    optimizer_g,
    loss_fn,
    device=device,
    amp=False,
    discriminator=discriminator,
    optimizer_d=optimizer_d,
    return_metrics=True,
    metric_normalization=ctx.validation.get("sr_metric_normalization", "dataset_stats"),
    metric_data_range=tuple(ctx.validation.get("sr_data_range", [-3.0, 3.0])),
)
assert all(math.isfinite(float(value)) for value in train_metrics.values())
print(f"training_startup=ok metrics={train_metrics}")
""",
}


def parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thebe-crops",
        type=Path,
        default=home / "Downloads" / "Thebe_Clean_Crops_V2",
    )
    parser.add_argument(
        "--faultseg-validation",
        type=Path,
        default=home / "Downloads" / "faultseg3d_val" / "validation",
    )
    parser.add_argument(
        "--swin-checkpoint",
        type=Path,
        default=home / "Downloads" / "swinunetr_tiny_after_faultseg3d.pth",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        help="Notebook filename to execute; repeat to select multiple notebooks.",
    )
    parser.add_argument("--keep-workdir", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    workdir = Path(tempfile.mkdtemp(prefix="sfr-notebook-smoke-", dir="/tmp"))
    results: list[dict[str, Any]] = []

    try:
        prepare_smoke_project(repo_root, workdir, args)
        notebooks = sorted((workdir / "notebooks").glob("*.ipynb"))
        if args.notebook:
            selected = set(args.notebook)
            notebooks = [path for path in notebooks if path.name in selected]
            missing = selected.difference(path.name for path in notebooks)
            if missing:
                raise FileNotFoundError(f"Unknown notebooks: {', '.join(sorted(missing))}")
        for path in notebooks:
            result = execute_notebook(path, workdir, args.timeout)
            results.append(result)
            status = "PASS" if result["success"] else "FAIL"
            print(f"{status} {path.name} ({result['seconds']:.1f}s)", flush=True)
            if not result["success"]:
                print(result["error"], flush=True)
            cleanup_runtime_outputs(workdir)

        report_path = workdir / "notebook-smoke-report.json"
        report_path.write_text(
            json.dumps({"workdir": str(workdir), "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"report={report_path}")
        return 0 if all(result["success"] for result in results) else 1
    finally:
        if not args.keep_workdir and results and all(result["success"] for result in results):
            shutil.rmtree(workdir, ignore_errors=True)


def prepare_smoke_project(repo_root: Path, workdir: Path, args: argparse.Namespace) -> None:
    validate_inputs(args)
    (workdir / "notebooks").mkdir(parents=True)
    (workdir / "configs" / "experiments").mkdir(parents=True)
    (workdir / "runtime").mkdir()
    (workdir / "assets").mkdir()
    os.symlink(repo_root / "src", workdir / "src")

    for path in (repo_root / "notebooks").glob("*.ipynb"):
        target = workdir / "notebooks" / path.name
        shutil.copy2(path, target)
        replace_training_cell(target)

    seis_path, fault_path = select_thebe_pair(args.thebe_crops)
    faultseg_dirs = prepare_faultseg_pair(args.faultseg_validation, workdir / "faultseg")
    sr_checkpoint = create_sr_checkpoint(workdir / "assets" / "sr_checkpoint.pth", repo_root)

    base = yaml.safe_load((repo_root / "configs" / "datasphere.yaml").read_text(encoding="utf-8"))
    base["clearml"]["enabled"] = False
    base["runtime"]["device"] = "cpu"
    base["runtime"]["num_workers"] = 0
    base["reproducibility"]["num_workers"] = 0
    (workdir / "configs" / "datasphere.yaml").write_text(
        yaml.safe_dump(base, sort_keys=False),
        encoding="utf-8",
    )

    for source in sorted((repo_root / "configs" / "experiments").glob("*.yaml")):
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        data = config.setdefault("data", {})
        for key in ("train_seis", "val_seis", "test_seis"):
            data[key] = str(seis_path)
        for key in ("train_fault", "val_fault", "test_fault"):
            data[key] = str(fault_path)
        data.update(faultseg_dirs)
        data["thebe_dir"] = str(args.thebe_crops)
        data["faultseg3d_dir"] = str(workdir / "faultseg")

        runtime_dir = workdir / "runtime" / source.stem
        checkpoints = config.setdefault("checkpoints", {})
        checkpoints["input"] = ""
        checkpoints["segmentation_input"] = ""
        checkpoints["sr_input"] = ""
        checkpoints["output_dir"] = str(runtime_dir / "checkpoints")
        checkpoints["best"] = str(runtime_dir / "checkpoints" / "best_model.pth")
        checkpoints["latest"] = str(runtime_dir / "checkpoints" / "latest_checkpoint.pth")
        if source.name == "12_segmentation_validation.yaml":
            checkpoints["input"] = str(args.swin_checkpoint)
        if source.name == "13_end_to_end_sr_segmentation_validation.yaml":
            checkpoints["segmentation_input"] = str(args.swin_checkpoint)
            checkpoints["sr_input"] = str(sr_checkpoint)

        outputs = config.setdefault("outputs", {})
        outputs["output_dir"] = str(runtime_dir / "outputs")

        training = config.setdefault("training", {})
        training.update(
            {
                "roi_size": [128, 128, 128],
                "patch_size": [128, 128, 128],
                "batch_size": 1,
                "max_epochs": 1,
                "amp": False,
                "num_workers": 0,
                "accumulation_steps": 1,
                "audit_max_arrays": 1,
            }
        )

        target = workdir / "configs" / "experiments" / source.name
        target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        if source.name == "16_3d_visualization.yaml":
            output_dir = Path(outputs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            target_mask = np.zeros((128, 128, 128), dtype=np.uint8)
            target_mask[32:96, 63:65, 32:96] = 1
            prediction = target_mask.copy()
            prediction[48:80, 80:82, 48:80] = 1
            np.save(output_dir / "preds.npy", prediction)
            np.save(output_dir / "gt.npy", target_mask)


def validate_inputs(args: argparse.Namespace) -> None:
    required = (
        args.thebe_crops / "seis",
        args.thebe_crops / "fault",
        args.faultseg_validation / "seis",
        args.faultseg_validation / "fault",
        args.swin_checkpoint,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing smoke-test inputs:\n" + "\n".join(missing))


def select_thebe_pair(root: Path) -> tuple[Path, Path]:
    seis_files = {path.name: path for path in (root / "seis").glob("*.npz")}
    fault_files = {path.name: path for path in (root / "fault").glob("*.npz")}
    common = sorted(set(seis_files).intersection(fault_files))
    if not common:
        raise ValueError(f"No matching Thebe crop names found below {root}")
    name = common[0]
    return seis_files[name], fault_files[name]


def prepare_faultseg_pair(source: Path, target: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split in ("train", "validation"):
        for kind in ("seis", "fault"):
            source_files = sorted((source / kind).glob("*.dat"))
            if not source_files:
                raise ValueError(f"No FaultSeg3D {kind} files found in {source / kind}")
            directory = target / split / kind
            directory.mkdir(parents=True, exist_ok=True)
            os.symlink(source_files[0], directory / "0.dat")
            key = f"faultseg3d_{'train' if split == 'train' else 'val'}_{kind}_dir"
            mapping[key] = str(directory)
    mapping["faultseg3d_shape"] = [128, 128, 128]
    mapping["faultseg3d_dtype"] = "float32"
    return mapping


def create_sr_checkpoint(path: Path, repo_root: Path) -> Path:
    sys.path.insert(0, str(repo_root / "src"))
    from seismic_fault_recognition.models.factory import build_model_by_name
    from seismic_fault_recognition.training import save_checkpoint

    model = build_model_by_name("omniseis_sr_generator")
    save_checkpoint(path, model, epoch=0, metrics={})
    return path


def replace_training_cell(path: Path) -> None:
    source = TRAINING_STARTUP_CELLS.get(path.name)
    if source is None:
        return
    notebook = nbformat.read(path, as_version=4)
    for cell in reversed(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{path.name}:original-training-cell", "exec")
            cell.source = source.strip()
            cell.outputs = []
            cell.execution_count = None
            nbformat.write(notebook, path)
            return
    raise ValueError(f"No code cell found in training notebook {path.name}")


def execute_notebook(path: Path, workdir: Path, timeout: int) -> dict[str, Any]:
    started = time.time()
    notebook = nbformat.read(path, as_version=4)
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"
    os.environ["IPYTHONDIR"] = str(workdir / ".ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(workdir / ".jupyter-runtime")
    os.environ["MPLCONFIGDIR"] = str(workdir / ".matplotlib")
    try:
        NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(workdir)}},
            allow_errors=False,
        ).execute()
        return {"notebook": path.name, "success": True, "seconds": time.time() - started}
    except CellExecutionError as exc:
        return {
            "notebook": path.name,
            "success": False,
            "seconds": time.time() - started,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "notebook": path.name,
            "success": False,
            "seconds": time.time() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def cleanup_runtime_outputs(workdir: Path) -> None:
    runtime = workdir / "runtime"
    for checkpoint_dir in runtime.glob("*/checkpoints"):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
