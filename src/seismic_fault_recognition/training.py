"""Training facade and shared checkpoint/metric helpers.

Stage-specific losses and loops live in ``losses.py`` and ``trainers.py``.
This module keeps backward-compatible imports plus generic utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

from .losses import (
    BinaryDiceLoss,
    BinaryDiceLossSymmetric,
    BinaryFocalLoss,
    CombinedLoss,
    build_loss,
    get_loss_profile,
    list_loss_profiles,
)
from .clearml import (
    ClearMLRun,
    clearml_metric_logger,
    init_clearml_from_context,
    init_clearml_task,
    report_checkpoint_artifact,
    report_optimizer_lr,
)
from .metrics import psnr_3d, segmentation_metrics_from_logits, sr_quality_metrics, ssim_3d
from .checkpoints import clean_state_dict
from .trainers import (
    get_trainer_profile,
    list_trainer_profiles,
    train_faultseg3d_epoch,
    train_simmim_epoch,
    train_sr_epoch,
    train_thebe_finetune_epoch,
    validate_segmentation_only,
    validate_sr_segmentation_pipeline,
    validate_simmim_reconstruction,
    validate_sr,
    validate_thebe_finetune,
)


def binary_metrics_np(prediction: Any, target: Any, threshold: float = 0.5) -> dict[str, float]:
    """Compute binary precision, recall, F1 and Dice with NumPy arrays."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("binary_metrics_np requires numpy") from exc

    pred = np.asarray(prediction) >= threshold
    tgt = np.asarray(target).astype(bool)
    tp = float(np.logical_and(pred, tgt).sum())
    fp = float(np.logical_and(pred, ~tgt).sum())
    fn = float(np.logical_and(~pred, tgt).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    return {"precision": precision, "recall": recall, "f1": f1, "dice": dice}


def save_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Save model weights and optional optimizer/scheduler state.

    Args:
        path: Target checkpoint path.
        model: PyTorch model or data-parallel wrapper.
        optimizer: Optional optimizer whose state should be included.
        scheduler: Optional scheduler whose state should be included.
        epoch: Optional epoch number.
        metrics: Optional flat checkpoint metrics.
        extra: Optional additional checkpoint fields.
    """

    torch = _require_torch()
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    state: dict[str, Any] = {"model_state_dict": model_state, "epoch": epoch, "metrics": metrics or {}}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        state.update(dict(extra))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    map_location: str | Any = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore model/optimizer/scheduler state.

    Args:
        path: Checkpoint file path.
        model: Optional model to receive the checkpoint state dict.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        map_location: PyTorch map location.
        strict: Strictness passed to ``model.load_state_dict``.

    Returns:
        Loaded checkpoint payload.
    """

    torch = _require_torch()
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if model is not None:
        model.load_state_dict(clean_state_dict(state_dict), strict=strict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def save_history(path: str | Path, history: list[dict[str, Any]]) -> None:
    """Save training history records as formatted JSON."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def train_one_epoch(*args: Any, **kwargs: Any) -> Any:
    """Backward-compatible alias for the Thebe fine-tune trainer."""

    return train_thebe_finetune_epoch(*args, **kwargs)


def validate_segmentation(*args: Any, **kwargs: Any) -> Any:
    """Backward-compatible alias for validation-only segmentation."""

    return validate_segmentation_only(*args, **kwargs)


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return clean_state_dict(state_dict)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Checkpoint helpers require PyTorch") from exc
    return torch
