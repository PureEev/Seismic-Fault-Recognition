"""Optional ClearML integration for training notebooks.

The helpers in this module are deliberately no-op friendly. Importing the
project or running notebooks locally must not require ClearML credentials; when
``clearml`` is unavailable, logging calls simply return without side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping
import inspect
import numbers


@dataclass
class ClearMLRun:
    """Small wrapper around a ClearML task/logger pair."""

    task: Any | None
    logger: Any | None
    enabled: bool
    project_name: str
    task_name: str
    reason: str = ""
    strict: bool = False

    def summary(self) -> str:
        """Return a one-line status string for logs and notebooks."""

        if self.enabled:
            return f"ClearML enabled: project={self.project_name!r}, task={self.task_name!r}"
        suffix = f" ({self.reason})" if self.reason else ""
        return f"ClearML disabled{suffix}"

    def connect(self, config: Mapping[str, Any] | Any) -> None:
        """Attach configuration to the ClearML task when logging is enabled."""

        if not self.enabled or self.task is None:
            return
        self._safe_call(self.task.connect, _to_plain(config))

    def report_scalar(self, title: str, series: str, value: Any, iteration: int | None) -> None:
        """Report one scalar metric if it can be converted to ``float``."""

        if not self.enabled or self.logger is None:
            return
        scalar = _to_float(value)
        if scalar is None:
            return
        self._safe_call(
            self.logger.report_scalar,
            title=str(title),
            series=str(series),
            value=scalar,
            iteration=0 if iteration is None else int(iteration),
        )

    def report_metrics(
        self,
        metrics: Mapping[str, Any] | Any,
        iteration: int | None,
        title: str = "Metrics",
        series_prefix: str = "",
    ) -> None:
        """Flatten and report a nested metric mapping to ClearML."""

        for key, value in flatten_metrics(metrics).items():
            metric_title = _title_for_metric(key, default=title)
            series = f"{series_prefix}/{key}" if series_prefix else key
            self.report_scalar(metric_title, series, value, iteration)

    def report_text(self, text: str, title: str = "Log", series: str = "Text", iteration: int | None = 0) -> None:
        """Report a text message through ClearML when the logger supports it."""

        if not self.enabled or self.logger is None:
            return
        if not hasattr(self.logger, "report_text"):
            return
        self._safe_call(self.logger.report_text, str(text), title=title, series=series, iteration=iteration or 0)

    def upload_artifact(self, name: str, artifact_path: str | Path) -> None:
        """Upload a local artifact path to the active ClearML task."""

        if not self.enabled or self.task is None or not hasattr(self.task, "upload_artifact"):
            return
        path = Path(artifact_path)
        if path.exists():
            self._safe_call(self.task.upload_artifact, name=name, artifact_object=str(path))

    def close(self) -> None:
        """Close the ClearML task if it was initialized."""

        if not self.enabled or self.task is None or not hasattr(self.task, "close"):
            return
        self._safe_call(self.task.close)

    def _safe_call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception:
            if self.strict:
                raise
            return None


def init_clearml_task(
    project_name: str,
    task_name: str,
    config: Mapping[str, Any] | Any | None = None,
    *,
    enabled: bool = True,
    tags: list[str] | tuple[str, ...] | None = None,
    output_uri: str | None = None,
    strict: bool = False,
) -> ClearMLRun:
    """Create a ClearML task when available, otherwise return a no-op run."""

    if not enabled:
        return ClearMLRun(None, None, False, project_name, task_name, reason="disabled in config", strict=strict)
    try:
        from clearml import Task  # type: ignore
    except ImportError:
        return ClearMLRun(None, None, False, project_name, task_name, reason="clearml is not installed", strict=strict)

    try:
        kwargs: dict[str, Any] = {"project_name": project_name, "task_name": task_name}
        if output_uri:
            kwargs["output_uri"] = output_uri
        task = Task.init(**kwargs)
        if tags and hasattr(task, "add_tags"):
            task.add_tags(list(tags))
        run = ClearMLRun(task, task.get_logger(), True, project_name, task_name, strict=strict)
        if config is not None:
            run.connect(config)
        return run
    except Exception as exc:
        if strict:
            raise
        return ClearMLRun(None, None, False, project_name, task_name, reason=str(exc), strict=strict)


def init_clearml_from_context(
    ctx: Any,
    *,
    enabled: bool | None = None,
    strict: bool = False,
    extra_config: Mapping[str, Any] | None = None,
) -> ClearMLRun:
    """Initialize ClearML using a ``NotebookContext`` and recipe metadata."""

    clearml_cfg = dict(getattr(ctx, "clearml", {}) or {})
    recipe = getattr(ctx, "recipe", None)
    stage = getattr(recipe, "stage", "")
    project_name = clearml_cfg.get("project_name") or _project_for_stage(stage)
    task_name = clearml_cfg.get("task_name") or getattr(recipe, "clearml_task_name", "Seismic_Experiment")
    is_enabled = bool(clearml_cfg.get("enabled", True) if enabled is None else enabled)

    payload: dict[str, Any] = {
        "recipe": recipe.as_dict() if hasattr(recipe, "as_dict") else _to_plain(recipe),
        "experiment": getattr(ctx, "exp_config", {}),
        "training": getattr(ctx, "training", {}),
        "data": getattr(ctx, "data", {}),
        "checkpoints": getattr(ctx, "checkpoints", {}),
        "outputs": getattr(ctx, "outputs", {}),
    }
    if extra_config:
        payload["extra"] = dict(extra_config)

    tags = [value for value in (stage, getattr(recipe, "dataset", ""), getattr(recipe, "model_variant", "")) if value]
    return init_clearml_task(
        project_name=project_name,
        task_name=task_name,
        config=payload,
        enabled=is_enabled,
        tags=tags,
        output_uri=clearml_cfg.get("output_uri"),
        strict=strict,
    )


def clearml_metric_logger(
    run: ClearMLRun | None,
    *,
    title: str = "Metrics",
    series_prefix: str = "",
    result_name: str = "loss",
    iteration_arg: str = "epoch",
    enabled: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate train/validation functions and log returned metrics.

    The wrapped function may return a scalar, a mapping, or an object with an
    ``item()`` method. Use ``clearml_iteration=epoch`` in calls when the
    wrapped function itself does not accept an epoch argument.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap one callable with optional metric logging."""

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Call the wrapped function and report returned metrics."""

            clearml_iteration = kwargs.pop("clearml_iteration", None)
            clearml_extra_metrics = kwargs.pop("clearml_extra_metrics", None)
            clearml_title = kwargs.pop("clearml_title", title)
            clearml_series_prefix = kwargs.pop("clearml_series_prefix", series_prefix)

            if clearml_iteration is None:
                clearml_iteration = _infer_iteration(fn, args, kwargs, iteration_arg)

            result = fn(*args, **kwargs)
            if enabled and run is not None:
                metrics = metrics_from_result(result, default_name=result_name)
                if clearml_extra_metrics:
                    metrics.update(flatten_metrics(clearml_extra_metrics))
                run.report_metrics(metrics, clearml_iteration, title=clearml_title, series_prefix=clearml_series_prefix)
            return result

        return wrapper

    return decorator


def report_optimizer_lr(run: ClearMLRun | None, optimizer: Any, iteration: int, series: str = "LR") -> None:
    """Report optimizer learning rates for all parameter groups."""

    if run is None or optimizer is None:
        return
    for index, group in enumerate(getattr(optimizer, "param_groups", []) or []):
        suffix = series if index == 0 else f"{series}/group_{index}"
        run.report_scalar("Learning_Rate", suffix, group.get("lr"), iteration)


def report_checkpoint_artifact(run: ClearMLRun | None, name: str, path: str | Path) -> None:
    """Upload a checkpoint artifact when ClearML logging is active."""

    if run is not None:
        run.upload_artifact(name, path)


def metrics_from_result(result: Any, default_name: str = "loss") -> dict[str, float]:
    """Convert a scalar or mapping return value into flat numeric metrics."""

    if isinstance(result, Mapping):
        return flatten_metrics(result)
    scalar = _to_float(result)
    return {default_name: scalar} if scalar is not None else {}


def flatten_metrics(metrics: Mapping[str, Any] | Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested metrics and drop non-numeric values."""

    if not isinstance(metrics, Mapping):
        scalar = _to_float(metrics)
        return {prefix or "value": scalar} if scalar is not None else {}
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_metrics(value, name))
            continue
        scalar = _to_float(value)
        if scalar is not None:
            flat[name] = scalar
    return flat


def _infer_iteration(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any], name: str) -> int | None:
    if name in kwargs:
        return int(kwargs[name])
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    except Exception:
        return None
    value = bound.arguments.get(name)
    return int(value) if value is not None else None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    return None


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _title_for_metric(name: str, default: str) -> str:
    lower = name.lower()
    if "lr" == lower or lower.endswith(".lr") or "learning_rate" in lower:
        return "Learning_Rate"
    if "loss" in lower:
        if lower.startswith("loss_") or ".loss_" in lower:
            return "Loss_Components"
        return "Loss"
    if any(token in lower for token in ("f1", "dice", "iou", "ap", "precision", "recall", "psnr", "ssim")):
        return "Metrics"
    return default


def _project_for_stage(stage: str) -> str:
    if stage == "simmim_pretrain":
        return "Seismic_Pretraining"
    if stage == "sr_training":
        return "Seismic_Super_Resolution"
    return "Seismic_Fault_Segmentation"
