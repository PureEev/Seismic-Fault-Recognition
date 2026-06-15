"""Checkpoint compatibility and diagnostics helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LoadStateDictResult:
    """Result of strict or loose model state loading.

    Attributes:
        success: Whether any load attempt succeeded.
        strict: Whether the successful attempt used strict loading.
        missing_keys: Model keys not found in the checkpoint.
        unexpected_keys: Checkpoint keys not consumed by the model.
        error: Error text from the failed strict/fallback attempt, if any.
    """

    success: bool
    strict: bool
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


@dataclass(frozen=True)
class SwinUNETRCheckpointInfo:
    """Detected SwinUNETR compatibility clues from checkpoint tensors."""

    total_tensors: int
    has_patch_embed: bool
    patch_kernel: tuple[int, ...] | None
    in_channels: int | None
    feature_size: int | None
    out_channels: int | None
    has_patch5: bool
    has_layer4: bool
    has_deep_cnn: bool
    has_adapters: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


@dataclass(frozen=True)
class BlockWeightStats:
    """Aggregated statistics for weights whose keys belong to one block."""

    block: str
    l2_norm: float
    sparsity_pct: float
    num_weights: int
    energy_fraction: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


def clean_state_dict(
    state_dict: Mapping[str, Any],
    strip_prefixes: Sequence[str] = ("module.",),
) -> dict[str, Any]:
    """Return a copy of ``state_dict`` with known wrapper prefixes removed."""

    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        for prefix in strip_prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def extract_state_dict(checkpoint_or_state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the model state dict from common checkpoint payload shapes."""

    for key in ("model_state_dict", "state_dict"):
        nested = checkpoint_or_state.get(key)
        if isinstance(nested, Mapping):
            return nested
    return checkpoint_or_state


def load_checkpoint_payload(path: str | Path, map_location: str | Any = "cpu") -> Mapping[str, Any]:
    """Load a checkpoint file and verify that it contains a mapping payload.

    Args:
        path: Path to a ``.pth``/``.pt`` checkpoint.
        map_location: PyTorch map location passed to ``torch.load``.

    Returns:
        Checkpoint dictionary as loaded by PyTorch.

    Raises:
        ImportError: If PyTorch is unavailable.
        TypeError: If the checkpoint payload is not mapping-like.
    """

    torch = _require_torch()
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected checkpoint mapping in {path}, got {type(payload)!r}")
    return payload


def load_state_dict_loose(
    model: Any,
    checkpoint_or_state: str | Path | Mapping[str, Any],
    map_location: str | Any = "cpu",
    strip_prefixes: Sequence[str] = ("module.",),
) -> LoadStateDictResult:
    """Try strict load first, then fall back to ``strict=False``."""

    if isinstance(checkpoint_or_state, (str, Path)):
        payload = load_checkpoint_payload(checkpoint_or_state, map_location=map_location)
    else:
        payload = checkpoint_or_state
    state_dict = clean_state_dict(extract_state_dict(payload), strip_prefixes=strip_prefixes)
    try:
        result = model.load_state_dict(state_dict, strict=True)
        return _load_result(success=True, strict=True, result=result)
    except RuntimeError as strict_error:
        try:
            result = model.load_state_dict(state_dict, strict=False)
            return _load_result(success=True, strict=False, result=result, error=str(strict_error))
        except RuntimeError as fallback_error:
            return LoadStateDictResult(
                success=False,
                strict=False,
                missing_keys=(),
                unexpected_keys=(),
                error=str(fallback_error),
            )


def inspect_swinunetr_checkpoint(checkpoint_or_state: Mapping[str, Any]) -> SwinUNETRCheckpointInfo:
    """Detect SwinUNETR variant clues from state dict keys and tensor shapes."""

    state_dict = clean_state_dict(extract_state_dict(checkpoint_or_state))
    patch_weight = state_dict.get("swinViT.patch_embed.proj.weight")
    patch_shape = _shape_tuple(patch_weight)
    out_shape = _shape_tuple(
        _first_existing(
            state_dict,
            (
                "out.conv.conv.weight",
                "out.conv.weight",
                "out.weight",
                "head.weight",
            ),
        )
    )
    patch_kernel = patch_shape[2:] if patch_shape is not None and len(patch_shape) >= 5 else None
    return SwinUNETRCheckpointInfo(
        total_tensors=len(state_dict),
        has_patch_embed=patch_shape is not None,
        patch_kernel=patch_kernel,
        in_channels=patch_shape[1] if patch_shape is not None and len(patch_shape) > 1 else None,
        feature_size=patch_shape[0] if patch_shape is not None else None,
        out_channels=out_shape[0] if out_shape is not None else None,
        has_patch5=patch_kernel == (5, 5, 5),
        has_layer4=any("swinViT.layers4" in key for key in state_dict),
        has_deep_cnn=any(key.startswith("encoder10") or key.startswith("decoder5") for key in state_dict),
        has_adapters="enc2to3.weight" in state_dict or "enc3to4.weight" in state_dict,
    )


def summarize_block_weights(
    checkpoint_or_state: Mapping[str, Any],
    blocks: Sequence[str] = (
        "encoder1",
        "encoder2",
        "encoder3",
        "encoder4",
        "enc2to3",
        "enc3to4",
        "decoder4",
        "decoder3",
        "decoder2",
        "decoder1",
        "out",
        "head",
    ),
    zero_threshold: float = 1e-3,
) -> list[BlockWeightStats]:
    """Compute per-block L2 energy and near-zero sparsity for floating weights."""

    state_dict = clean_state_dict(extract_state_dict(checkpoint_or_state))
    raw_stats: list[tuple[str, float, float, int]] = []
    total_energy = 0.0
    for block in blocks:
        arrays = [
            _to_numpy(value).reshape(-1)
            for key, value in state_dict.items()
            if key.startswith(block) and "weight" in key and _is_numeric_tensor(value)
        ]
        if not arrays:
            continue
        values = np.concatenate(arrays).astype(np.float64, copy=False)
        l2_norm = float(np.linalg.norm(values, ord=2))
        sparsity_pct = float(np.mean(np.abs(values) < float(zero_threshold)) * 100.0)
        num_weights = int(values.size)
        raw_stats.append((block, l2_norm, sparsity_pct, num_weights))
        total_energy += l2_norm
    return [
        BlockWeightStats(
            block=block,
            l2_norm=l2_norm,
            sparsity_pct=sparsity_pct,
            num_weights=num_weights,
            energy_fraction=(l2_norm / total_energy if total_energy > 0 else 0.0),
        )
        for block, l2_norm, sparsity_pct, num_weights in raw_stats
    ]


def _load_result(
    success: bool,
    strict: bool,
    result: Any,
    error: str = "",
) -> LoadStateDictResult:
    missing = tuple(str(key) for key in getattr(result, "missing_keys", ()))
    unexpected = tuple(str(key) for key in getattr(result, "unexpected_keys", ()))
    return LoadStateDictResult(
        success=success,
        strict=strict,
        missing_keys=missing,
        unexpected_keys=unexpected,
        error=error,
    )


def _first_existing(state_dict: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in state_dict:
            return state_dict[key]
    return None


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def _is_numeric_tensor(value: Any) -> bool:
    if hasattr(value, "is_floating_point"):
        try:
            return bool(value.is_floating_point())
        except TypeError:
            pass
    try:
        return np.issubdtype(np.asarray(value).dtype, np.number)
    except TypeError:
        return False


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional for tests.
        raise ImportError("Loading checkpoint files requires PyTorch") from exc
    return torch
