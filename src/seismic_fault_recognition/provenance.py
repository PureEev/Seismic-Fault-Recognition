"""Experiment and data provenance helpers kept outside checkpoint payloads."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import subprocess


DEFAULT_PACKAGES = (
    "numpy",
    "torch",
    "torchvision",
    "monai",
    "scipy",
    "clearml",
    "pyyaml",
)


def build_data_manifest(
    paths: Mapping[str, str | Path] | Sequence[str | Path],
    split_name: str | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Describe local data files without downloading or mutating them."""

    items = paths.items() if isinstance(paths, Mapping) else ((str(index), path) for index, path in enumerate(paths))
    files = []
    for name, raw_path in items:
        if raw_path is None or str(raw_path) == "":
            continue
        path = _resolve_path(raw_path, base_dir)
        record: dict[str, Any] = {
            "name": str(name),
            "path": str(path),
            "exists": path.exists(),
            "suffix": path.suffix,
        }
        if path.exists():
            stat = path.stat()
            record["size_bytes"] = int(stat.st_size)
            if path.suffix == ".npz":
                record["arrays"] = _describe_npz(path)
        files.append(record)
    return {"split_name": split_name, "files": files}


def save_experiment_manifest(
    ctx: Any,
    data_manifest: Mapping[str, Any],
    output_path: str | Path,
    packages: Sequence[str] = DEFAULT_PACKAGES,
) -> Path:
    """Save recipe/config/data/version metadata as JSON outside the checkpoint."""

    repo_root = Path(getattr(ctx, "repo_root", Path.cwd()))
    recipe = getattr(ctx, "recipe", None)
    payload = {
        "recipe": recipe.as_dict() if hasattr(recipe, "as_dict") else _to_plain(recipe),
        "base_config": _to_plain(getattr(ctx, "base_config", None)),
        "experiment_config": _to_plain(getattr(ctx, "exp_config", {})),
        "data_manifest": _to_plain(data_manifest),
        "git_sha": git_sha(repo_root),
        "packages": package_versions(packages),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def package_versions(packages: Sequence[str] = DEFAULT_PACKAGES) -> dict[str, str]:
    """Return installed versions for requested packages."""

    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def git_sha(repo_root: str | Path) -> str:
    """Return the current Git commit SHA for ``repo_root`` if available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _describe_npz(path: Path) -> list[dict[str, Any]]:
    try:
        import numpy as np
    except ImportError:
        return [{"error": "numpy is not installed"}]

    rows: list[dict[str, Any]] = []
    with np.load(path, mmap_mode="r") as data:
        for key in data.files:
            arr = data[key]
            rows.append(
                {
                    "key": str(key),
                    "shape": [int(item) for item in arr.shape],
                    "dtype": str(arr.dtype),
                    "ndim": int(arr.ndim),
                }
            )
    return rows


def _resolve_path(path: str | Path, base_dir: str | Path | None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or base_dir is None:
        return candidate
    return Path(base_dir).expanduser() / candidate


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
