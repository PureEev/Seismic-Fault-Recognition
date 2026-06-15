"""Stage-specific training loops and trainer profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
from seismic_fault_recognition.registry import TRAINER_REGISTRY


@dataclass(frozen=True)
class TrainerProfile:
    """Registered training loop metadata.

    Attributes:
        name: Stable registry key.
        stage: Experiment stage that uses the profile.
        train_fn: One-epoch training function.
        validate_fn: Full validation function.
        description: Human-readable documentation string.
    """

    name: str
    stage: str
    train_fn: Callable[..., Any] | None
    validate_fn: Callable[..., Any] | None
    description: str


def list_trainer_profiles() -> tuple[str, ...]:
    """Return registered trainer profile names."""
    return tuple(TRAINER_REGISTRY.list())


def get_trainer_profile(name: str) -> TrainerProfile:
    """Return a registered trainer profile by name."""
    return TRAINER_REGISTRY.get(name)


def normalize_batch_dims(inputs: Any, targets: Any | None = None) -> tuple[Any, Any | None]:
    """Normalize notebook batch shapes to ``B,C,D,H,W`` where possible."""

    if hasattr(inputs, "dim") and inputs.dim() == 4:
        inputs = inputs.unsqueeze(1)
    if targets is not None and hasattr(targets, "dim") and targets.dim() == 4:
        targets = targets.unsqueeze(1)
    return inputs, targets


def train_faultseg3d_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: str | Any = "cuda",
    amp: bool = True,
) -> float:
    """Run one FaultSeg3D training epoch with optional AMP."""

    import torch
    from tqdm import tqdm

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp))
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train (FaultSeg3D)", leave=False):
        images, labels = _unpack_segmentation_batch(batch)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=bool(amp)):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
    return total_loss / len(loader)


def evaluate_faultseg3d(
    model: Any,
    loader: Any,
    loss_fn: Any,
    device: str | Any = "cuda",
    threshold: float = 0.5,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate FaultSeg3D segmentation on one validation split."""

    import torch
    from tqdm import tqdm
    from .metrics import segmentation_metrics_from_logits

    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval (FaultSeg3D)", leave=False):
            images, labels = _unpack_segmentation_batch(batch)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += float(loss_fn(outputs, labels).item())
            all_logits.append(outputs.cpu())
            all_targets.append(labels.cpu())

    metrics = segmentation_metrics_from_logits(
        torch.cat(all_logits),
        torch.cat(all_targets),
        thresholds=thresholds or (threshold,),
        default_threshold=threshold,
    )
    metrics["loss"] = total_loss / len(loader)
    return {**metrics, **_alias_val_metrics(metrics)}


def train_thebe_finetune_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: str | Any = "cuda",
    amp: bool = True,
    accumulation_steps: int = 1,
) -> float:
    """Run one Thebe fine-tuning epoch with optional gradient accumulation."""

    import torch
    from tqdm import tqdm

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp))
    total_loss = 0.0
    optimizer.zero_grad()
    for i, (images, labels) in enumerate(tqdm(loader, desc="Train (Thebe)", leave=False)):
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=bool(amp)):
            outputs = model(images)
            loss = loss_fn(outputs, labels) / accumulation_steps
        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += float(loss.item()) * accumulation_steps
    return total_loss / len(loader)


def validate_thebe_finetune(
    model: Any,
    loader: Any,
    loss_fn: Any,
    device: str | Any = "cuda",
    threshold: float = 0.5,
    thresholds: Sequence[float] | None = None,
) -> dict[str, float]:
    """Validate Thebe segmentation and return threshold-independent metrics."""

    import torch
    from tqdm import tqdm
    from .metrics import segmentation_metrics_from_logits

    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val (Thebe)", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += float(loss_fn(logits, labels).item())
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    metrics = segmentation_metrics_from_logits(
        torch.cat(all_logits),
        torch.cat(all_targets),
        thresholds=thresholds or (threshold,),
    )
    metrics["loss"] = total_loss / len(loader)
    return {**metrics, **_alias_val_metrics(metrics)}


def train_simmim_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: str | Any = "cuda",
    amp: bool = True,
) -> float:
    """Run one SimMIM masked reconstruction epoch."""

    import torch
    from tqdm import tqdm

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp))
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train (SimMIM)", leave=False):
        masked, target, masks = _unpack_simmim_batch(batch)
        masked, target, masks = masked.to(device), target.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=bool(amp)):
            outputs = model(masked)
            loss = masked_simmim_l1_loss(outputs, target, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
    return total_loss / len(loader)


def validate_simmim_reconstruction(
    model: Any,
    loader: Any,
    loss_fn: Any,
    device: str | Any = "cuda",
) -> dict[str, float]:
    """Validate masked reconstruction performance."""

    import torch
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val (SimMIM)", leave=False):
            masked, target, masks = _unpack_simmim_batch(batch)
            masked, target, masks = masked.to(device), target.to(device), masks.to(device)
            outputs = model(masked)
            total_loss += float(masked_simmim_l1_loss(outputs, target, masks).item())
    loss = total_loss / len(loader)
    return {"loss": loss, "val_loss": loss}


def masked_simmim_l1_loss(pred: Any, target: Any, mask: Any) -> Any:
    """Compute L1 loss only on masked seismic voxels."""

    import torch.nn.functional as F
    return F.l1_loss(pred * mask, target * mask, reduction="sum") / (mask.sum() + 1e-8)


def train_sr_epoch(
    gen: Any,
    loader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: str | Any = "cuda",
    amp: bool = True,
    discriminator: Any | None = None,
    optimizer_d: Any | None = None,
    return_metrics: bool = False,
    metric_normalization: str = "dataset_stats",
    metric_data_range: tuple[float, float] | None = (-3.0, 3.0),
) -> dict[str, float]:
    """Run one SR training epoch for GAN-based generator/discriminator."""

    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from .metrics import sr_quality_metrics

    gen.train()
    if discriminator is not None:
        discriminator.train()
    scaler_g = torch.cuda.amp.GradScaler(enabled=bool(amp))
    scaler_d = torch.cuda.amp.GradScaler(enabled=bool(amp))

    metrics = {"loss_g": 0.0, "loss_d": 0.0, "psnr": 0.0, "ssim": 0.0}
    for lr, hr in tqdm(loader, desc="Train (SR)", leave=False):
        lr, hr = lr.to(device), hr.to(device)

        loss_d = torch.zeros((), device=device)
        if discriminator is not None:
            if optimizer_d is None:
                raise ValueError("optimizer_d is required when discriminator is provided")
            optimizer_d.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(amp)):
                sr_detached = gen(lr).detach()
                d_real = discriminator(hr)
                d_fake = discriminator(sr_detached)
                loss_d = (
                    F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                    + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                ) * 0.5
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=bool(amp)):
            sr = gen(lr)
            d_fake = discriminator(sr) if discriminator is not None else None
            loss_g = loss_fn(sr, hr, disc_logits=d_fake)
        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer)
        scaler_g.update()

        quality = sr_quality_metrics(
            sr.detach(),
            hr,
            normalization=metric_normalization,
            data_range=metric_data_range,
        )
        metrics["loss_g"] += float(loss_g.item())
        metrics["loss_d"] += float(loss_d.item())
        metrics["psnr"] += quality["psnr"]
        metrics["ssim"] += quality["ssim"]

    averaged = {key: value / len(loader) for key, value in metrics.items()}
    return averaged if return_metrics else {"loss": averaged["loss_g"]}


def validate_sr(
    gen: Any,
    loader: Any,
    loss_fn: Any,
    device: str | Any = "cuda",
    metric_normalization: str = "dataset_stats",
    metric_data_range: tuple[float, float] | None = (-3.0, 3.0),
) -> dict[str, float]:
    """Validate SR generator using PSNR/SSIM and reconstruction loss."""

    import torch
    from tqdm import tqdm
    from .metrics import sr_quality_metrics

    gen.eval()
    total_loss = 0.0
    all_psnr, all_ssim = [], []
    with torch.no_grad():
        for lr, hr in tqdm(loader, desc="Val (SR)", leave=False):
            lr, hr = lr.to(device), hr.to(device)
            sr = gen(lr)
            total_loss += float(loss_fn(sr, hr).item())
            q = sr_quality_metrics(
                sr,
                hr,
                normalization=metric_normalization,
                data_range=metric_data_range,
            )
            all_psnr.append(q["psnr"])
            all_ssim.append(q["ssim"])

    import numpy as np
    metrics = {
        "loss": total_loss / len(loader),
        "psnr": float(np.mean(all_psnr)),
        "ssim": float(np.mean(all_ssim)),
    }
    return {**metrics, **{f"val_{key}": value for key, value in metrics.items()}}


def validate_segmentation_only(
    model: Any,
    loader: Any,
    loss_fn: Any,
    device: str | Any = "cuda",
    threshold: float = 0.5,
    thresholds: Sequence[float] | None = None,
) -> dict[str, float]:
    """Minimal validation loop for standalone segmentation metrics."""

    return validate_thebe_finetune(model, loader, loss_fn, device=device, threshold=threshold, thresholds=thresholds)


def validate_sr_segmentation_pipeline(
    sr_model: Any,
    seg_model: Any,
    loader: Any,
    loss_fn: Any,
    device: str | Any = "cuda",
    threshold: float = 0.5,
    thresholds: Sequence[float] | None = None,
) -> dict[str, float]:
    """Validate a chained SR + segmentation pipeline."""

    import torch
    from tqdm import tqdm
    from .metrics import segmentation_metrics_from_logits

    sr_model.eval()
    seg_model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []
    with torch.no_grad():
        for lr, hr_fault in tqdm(loader, desc="Val (SR+Seg)", leave=False):
            lr, hr_fault = lr.to(device), hr_fault.to(device)
            sr = sr_model(lr)
            logits = seg_model(sr)
            total_loss += float(loss_fn(logits, hr_fault).item())
            all_logits.append(logits.cpu())
            all_targets.append(hr_fault.cpu())

    metrics = segmentation_metrics_from_logits(
        torch.cat(all_logits),
        torch.cat(all_targets),
        thresholds=thresholds or (threshold,),
    )
    metrics["loss"] = total_loss / len(loader)
    return {**metrics, **_alias_val_metrics(metrics)}


def _alias_val_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    aliased = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            aliased.setdefault(f"val_{key}", float(value))
    return aliased


def _unpack_segmentation_batch(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, dict):
        images = batch.get("image", batch.get("seismic"))
        labels = batch.get("label", batch.get("fault"))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, labels = batch[0], batch[1]
    else:
        raise TypeError("Expected a segmentation batch mapping or (images, labels) sequence")
    if images is None or labels is None:
        raise ValueError("Segmentation batch must contain both images and labels")
    normalized_images, normalized_labels = normalize_batch_dims(images, labels)
    return normalized_images, normalized_labels


def _unpack_simmim_batch(batch: Any) -> tuple[Any, Any, Any]:
    if not isinstance(batch, (tuple, list)):
        raise TypeError("Expected a SimMIM batch sequence")
    if len(batch) == 3:
        masked, target, mask = batch
        return masked, target, mask
    if len(batch) == 2:
        images, mask = batch
        return images, images, mask
    raise ValueError("Expected SimMIM batch as (masked, target, mask) or (images, mask)")


# Registration
TRAINER_REGISTRY.register("faultseg3d_pretrain")(
    TrainerProfile(
        name="faultseg3d_pretrain",
        stage="faultseg3d_pretrain",
        train_fn=train_faultseg3d_epoch,
        validate_fn=evaluate_faultseg3d,
        description="AMP supervised pretrain loop with F1/AP-style validation.",
    )
)
TRAINER_REGISTRY.register("thebe_finetune")(
    TrainerProfile(
        name="thebe_finetune",
        stage="thebe_finetune",
        train_fn=train_thebe_finetune_epoch,
        validate_fn=validate_thebe_finetune,
        description="Thebe fine-tune loop with channel normalization and optional accumulation.",
    )
)
TRAINER_REGISTRY.register("simmim_pretrain")(
    TrainerProfile(
        name="simmim_pretrain",
        stage="simmim_pretrain",
        train_fn=train_simmim_epoch,
        validate_fn=validate_simmim_reconstruction,
        description="Masked reconstruction loop using masked regions for L1 loss.",
    )
)
TRAINER_REGISTRY.register("sr_training")(
    TrainerProfile(
        name="sr_training",
        stage="sr_training",
        train_fn=train_sr_epoch,
        validate_fn=validate_sr,
        description="Generator-focused SR loop with composite reconstruction loss.",
    )
)
TRAINER_REGISTRY.register("segmentation_validation")(
    TrainerProfile(
        name="segmentation_validation",
        stage="validation",
        train_fn=None,
        validate_fn=validate_segmentation_only,
        description="Validation-only loop for segmentation checkpoints.",
    )
)
