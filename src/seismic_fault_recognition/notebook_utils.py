"""Small helpers shared by runnable notebooks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import os
import random
import sys

from .config import ExperimentConfig, load_config
from .recipes import ExperimentRecipe, get_recipe, load_experiment_config


@dataclass
class NotebookContext:
    """Resolved package, config, recipe, and path context for notebooks."""

    repo_root: Path
    base_config: ExperimentConfig
    recipe: ExperimentRecipe
    exp_config: dict[str, Any]
    data: dict[str, Any]
    training: dict[str, Any]
    checkpoints: dict[str, Any]
    outputs: dict[str, Any]
    clearml: dict[str, Any]
    validation: dict[str, Any]
    reproducibility: dict[str, Any]


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upward from ``start``.

    Args:
        start: Optional path to start from. Defaults to the current directory.

    Returns:
        First parent containing both ``src`` and ``configs``, or the start path.
    """

    path = Path(start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    return path


def ensure_src_on_path(repo_root: str | Path) -> None:
    """Prepend ``repo_root/src`` to ``sys.path`` for notebook execution."""

    src_path = str(Path(repo_root) / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def load_notebook_context(recipe_name: str, start: str | Path | None = None) -> NotebookContext:
    """Load the standard context used by runnable experiment notebooks."""

    repo_root = find_repo_root(start)
    ensure_src_on_path(repo_root)
    base_config = load_config(repo_root / "configs" / "datasphere.yaml")
    recipe = get_recipe(recipe_name)
    exp_config = load_experiment_config(repo_root / "configs" / "experiments" / recipe.config_name)
    clearml = _build_clearml_config(base_config, recipe, exp_config)
    validation = _merge_config_section(base_config, exp_config, "validation")
    reproducibility = _merge_config_section(base_config, exp_config, "reproducibility")
    return NotebookContext(
        repo_root=repo_root,
        base_config=base_config,
        recipe=recipe,
        exp_config=exp_config,
        data=dict(exp_config.get("data", {})),
        training=dict(exp_config.get("training", {})),
        checkpoints=dict(exp_config.get("checkpoints", {})),
        outputs=dict(exp_config.get("outputs", {})),
        clearml=clearml,
        validation=validation,
        reproducibility=reproducibility,
    )


def resolve_path(value: str | Path, repo_root: str | Path) -> Path:
    """Resolve an absolute or repository-relative path."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(repo_root) / path


def resolve_many(values: Mapping[str, str | Path], repo_root: str | Path) -> dict[str, Path]:
    """Resolve a mapping of paths relative to the repository root."""

    return {key: resolve_path(value, repo_root) for key, value in values.items() if str(value)}


def print_path_report(paths: Mapping[str, str | Path], repo_root: str | Path) -> None:
    """Print existence status for a mapping of configured paths."""

    for key, value in resolve_many(paths, repo_root).items():
        status = "OK" if value.exists() else "missing"
        print(f"{key}: {value} -> {status}")


def ensure_dir(path: str | Path, repo_root: str | Path) -> Path:
    """Resolve and create a directory if needed."""

    resolved = resolve_path(path, repo_root)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def seed_everything(seed: int, deterministic: bool | None = None, benchmark: bool | None = None) -> None:
    """Seed Python, NumPy, and PyTorch RNGs when available."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic is not None:
            torch.backends.cudnn.deterministic = bool(deterministic)
        if benchmark is not None:
            torch.backends.cudnn.benchmark = bool(benchmark)
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python RNGs inside a PyTorch DataLoader worker."""

    worker_seed = 42 + int(worker_id)
    try:
        import torch

        worker_seed = int(torch.initial_seed() % 2**32)
    except ImportError:
        pass
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


def make_torch_generator(seed: int) -> Any | None:
    """Return a seeded PyTorch generator, or ``None`` when PyTorch is unavailable."""

    try:
        import torch
    except ImportError:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def dataloader_kwargs(
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    *,
    shuffle: bool | None = None,
) -> dict[str, Any]:
    """Common deterministic DataLoader kwargs for notebooks."""

    kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "worker_init_fn": seed_worker,
    }
    generator = make_torch_generator(seed)
    if generator is not None:
        kwargs["generator"] = generator
    if shuffle is not None:
        kwargs["shuffle"] = bool(shuffle)
    return kwargs


def tuple3(value: Sequence[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert a length-3 sequence to a tuple of ints."""

    return tuple(int(x) for x in value)  # type: ignore[return-value]


def print_context_summary(ctx: NotebookContext) -> None:
    """Print the most important recipe and validation settings."""

    print(f"repo_root: {ctx.repo_root}")
    print(f"recipe: {ctx.recipe.name}")
    print(f"stage: {ctx.recipe.stage}")
    print(f"dataset: {ctx.recipe.dataset}")
    print(f"model_variant: {ctx.recipe.model_variant}")
    print(f"loss_profile: {ctx.recipe.loss_profile}")
    print(f"trainer_profile: {ctx.recipe.trainer_profile}")
    print(f"clearml_project: {ctx.clearml.get('project_name')}")
    print(f"clearml_task: {ctx.clearml.get('task_name')}")
    print(f"primary_metric: {ctx.validation.get('primary_metric')}")


def _build_clearml_config(
    base_config: ExperimentConfig,
    recipe: ExperimentRecipe,
    exp_config: Mapping[str, Any],
) -> dict[str, Any]:
    clearml = asdict(base_config.clearml)
    stage_project_map = {
        "simmim_pretrain": clearml.get("pretraining_project_name", clearml.get("project_name")),
        "sr_training": clearml.get("sr_project_name", clearml.get("project_name")),
    }
    clearml["project_name"] = stage_project_map.get(
        recipe.stage,
        clearml.get("segmentation_project_name", clearml.get("project_name")),
    )
    clearml["task_name"] = recipe.clearml_task_name
    override = exp_config.get("clearml", {})
    if isinstance(override, Mapping):
        clearml.update(dict(override))
    return clearml


def _merge_config_section(
    base_config: ExperimentConfig,
    exp_config: Mapping[str, Any],
    section_name: str,
) -> dict[str, Any]:
    section = getattr(base_config, section_name, None)
    merged = asdict(section) if section is not None else {}
    override = exp_config.get(section_name, {})
    if isinstance(override, Mapping):
        merged.update(dict(override))
    return merged
