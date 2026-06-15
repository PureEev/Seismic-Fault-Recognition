"""Metric helpers shared by training and validation loops."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import math


def segmentation_metrics_from_logits(
    logits: Any,
    targets: Any,
    thresholds: Sequence[float] = (0.5,),
    default_threshold: float = 0.5,
) -> dict[str, Any]:
    """Binary segmentation metrics for logits shaped like ``B,C,D,H,W``."""

    torch, _ = _require_torch_modules()
    probs = torch.sigmoid(logits.float()).detach().reshape(-1).cpu()
    target = (targets.detach().reshape(-1).cpu() > 0.5)
    return segmentation_metrics_from_probabilities(
        probs,
        target,
        thresholds=thresholds,
        default_threshold=default_threshold,
    )


def segmentation_metrics_from_probabilities(
    probabilities: Any,
    targets: Any,
    thresholds: Sequence[float] = (0.5,),
    default_threshold: float = 0.5,
) -> dict[str, Any]:
    """Binary segmentation metrics from probabilities and binary targets."""

    torch, _ = _require_torch_modules()
    probs = probabilities.float().reshape(-1) if hasattr(probabilities, "float") else torch.as_tensor(probabilities).float().reshape(-1)
    target = targets.bool().reshape(-1) if hasattr(targets, "bool") else torch.as_tensor(targets).bool().reshape(-1)
    thresholds_clean = _normalize_thresholds(thresholds, default_threshold)
    per_threshold = {
        _threshold_key(threshold): _binary_metrics_at_threshold(probs, target, threshold)
        for threshold in thresholds_clean
    }
    default_key = _threshold_key(default_threshold)
    default_metrics = dict(per_threshold[default_key])
    best_key, best_metrics = max(
        per_threshold.items(),
        key=lambda item: (item[1]["dice"], item[1]["f1"], item[1]["recall"]),
    )
    ranking = _ranking_metrics(probs, target)
    result: dict[str, Any] = {
        **default_metrics,
        **ranking,
        f"f1@{_threshold_key(default_threshold)}": default_metrics["f1"],
        "best_threshold": float(best_key),
        "precision_best_threshold": best_metrics["precision"],
        "recall_best_threshold": best_metrics["recall"],
        "f1_best_threshold": best_metrics["f1"],
        "dice_best_threshold": best_metrics["dice"],
        "iou_best_threshold": best_metrics["iou"],
        "per_threshold": per_threshold,
    }
    return result


def psnr_3d(
    sr: Any,
    hr: Any,
    normalization: str = "dataset_stats",
    data_range: tuple[float, float] | None = (-3.0, 3.0),
) -> float:
    """PSNR for 3D tensors shaped as ``B,C,D,H,W``."""

    torch, _ = _require_torch_modules()
    sr_norm, hr_norm = _normalize_pair(sr.float(), hr.float(), normalization=normalization, data_range=data_range)
    mse = torch.mean((sr_norm - hr_norm) ** 2)
    if float(mse.detach().cpu()) <= 0.0:
        return 100.0
    return 10.0 * math.log10(1.0 / float(mse.detach().cpu()))


def ssim_3d(
    sr: Any,
    hr: Any,
    window_size: int = 11,
    sigma: float = 1.5,
    normalization: str = "dataset_stats",
    data_range: tuple[float, float] | None = (-3.0, 3.0),
) -> float:
    """SSIM for 3D tensors shaped as ``B,C,D,H,W``."""

    torch, F = _require_torch_modules()
    sr_norm, hr_norm = _normalize_pair(sr.float(), hr.float(), normalization=normalization, data_range=data_range)
    _, channels, depth, height, width = sr_norm.shape
    size = int(min(window_size, depth, height, width))
    if size % 2 == 0:
        size -= 1
    size = max(size, 3)

    coords = torch.arange(size, dtype=sr_norm.dtype, device=sr_norm.device)
    coords = coords - size // 2
    kernel_1d = torch.exp(-(coords**2) / float(2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    window = kernel_3d.expand(channels, 1, size, size, size).contiguous()

    padding = size // 2
    mu1 = F.conv3d(sr_norm, window, padding=padding, groups=channels)
    mu2 = F.conv3d(hr_norm, window, padding=padding, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv3d(sr_norm * sr_norm, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv3d(hr_norm * hr_norm, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv3d(sr_norm * hr_norm, window, padding=padding, groups=channels) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().detach().cpu())


def sr_quality_metrics(
    sr: Any,
    hr: Any,
    normalization: str = "dataset_stats",
    data_range: tuple[float, float] | None = (-3.0, 3.0),
) -> dict[str, float]:
    """Return PSNR and SSIM for super-resolution outputs.

    Args:
        sr: Super-resolved tensor shaped ``B,C,D,H,W``.
        hr: High-resolution reference tensor with matching shape.
        normalization: Normalization policy before metric computation.
        data_range: Fixed low/high range for ``dataset_stats`` or ``fixed_range``.

    Returns:
        Mapping with ``psnr`` and ``ssim`` floats.
    """

    return {
        "psnr": psnr_3d(sr, hr, normalization=normalization, data_range=data_range),
        "ssim": ssim_3d(sr, hr, normalization=normalization, data_range=data_range),
    }


def flatten_metric_mapping(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested numeric metrics into dot-separated keys."""

    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_metric_mapping(value, prefix=name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = float(value)
    return flat


def _minmax01(x: Any) -> Any:
    return (x - x.amin()) / (x.amax() - x.amin() + 1e-8)


def _normalize_pair(
    sr: Any,
    hr: Any,
    normalization: str,
    data_range: tuple[float, float] | None,
) -> tuple[Any, Any]:
    if normalization == "per_volume_minmax":
        return _minmax01(sr), _minmax01(hr)
    if normalization == "none":
        return sr, hr
    if normalization in {"dataset_stats", "fixed_range"}:
        low, high = data_range or (-3.0, 3.0)
        scale = float(high) - float(low)
        if scale <= 0:
            raise ValueError("data_range high value must be greater than low value")
        return _clip01((sr - float(low)) / scale), _clip01((hr - float(low)) / scale)
    raise ValueError(f"Unknown SR metric normalization: {normalization!r}")


def _clip01(x: Any) -> Any:
    return x.clamp(0.0, 1.0) if hasattr(x, "clamp") else x


def _normalize_thresholds(thresholds: Sequence[float], default_threshold: float) -> tuple[float, ...]:
    values = [float(item) for item in thresholds]
    values.append(float(default_threshold))
    clean = sorted({round(value, 6) for value in values if 0.0 <= float(value) <= 1.0})
    if not clean:
        return (float(default_threshold),)
    return tuple(float(item) for item in clean)


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.6g}"


def _binary_metrics_at_threshold(probs: Any, target: Any, threshold: float) -> dict[str, float]:
    pred = probs >= float(threshold)
    tp = float((pred & target).sum().item())
    fp = float((pred & ~target).sum().item())
    fn = float((~pred & target).sum().item())
    tn = float((~pred & ~target).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _ranking_metrics(probs: Any, target: Any) -> dict[str, float]:
    torch, _ = _require_torch_modules()
    positives = float(target.sum().item())
    if positives <= 0.0 or probs.numel() == 0:
        return {"ap": 0.0, "pr_auc": 0.0}
    order = torch.argsort(probs, descending=True)
    sorted_target = target[order].float()
    tp = torch.cumsum(sorted_target, dim=0)
    fp = torch.cumsum(1.0 - sorted_target, dim=0)
    recall = tp / max(positives, 1.0)
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall_with_origin = torch.cat([torch.zeros(1, dtype=recall.dtype, device=recall.device), recall])
    precision_with_origin = torch.cat([torch.ones(1, dtype=precision.dtype, device=precision.device), precision])
    delta_recall = recall_with_origin[1:] - recall_with_origin[:-1]
    ap = float((delta_recall * precision_with_origin[1:]).sum().item())
    trapz = getattr(torch, "trapezoid", torch.trapz)
    pr_auc = float(trapz(precision_with_origin, recall_with_origin).item())
    return {"ap": ap, "pr_auc": pr_auc}


def _require_torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise ImportError("3D quality metrics require PyTorch") from exc
    return torch, F
