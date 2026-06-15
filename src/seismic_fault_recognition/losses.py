"""Loss profiles for segmentation, reconstruction and SR workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from seismic_fault_recognition.registry import LOSS_REGISTRY


@dataclass(frozen=True)
class LossProfile:
    """Registered loss configuration for one training or validation stage."""

    name: str
    stage: str
    description: str
    weights: dict[str, float]
    factory: Callable[..., Any]


def list_loss_profiles() -> tuple[str, ...]:
    """Return registered loss profile names."""
    return tuple(LOSS_REGISTRY.list())


def get_loss_profile(name: str) -> LossProfile:
    """Return a registered loss profile by name (compatibility alias)."""
    # This matches the old behavior for tests.
    profiles = {
        "faultseg3d_combined_sym": LossProfile(
            name="faultseg3d_combined_sym",
            stage="faultseg3d_pretrain",
            description="BCE + symmetric Dice + Focal, weights 0.2/0.4/0.4.",
            weights={"bce": 0.2, "dice": 0.4, "focal": 0.4},
            factory=lambda **kw: make_combined_loss(w_bce=0.2, w_dice=0.4, w_focal=0.4, symmetric_dice=True, **kw),
        ),
        "thebe_combined_sym": LossProfile(
            name="thebe_combined_sym",
            stage="thebe_finetune",
            description="Thebe fine-tuning BCE + symmetric Dice + Focal, weights 0.2/0.4/0.4.",
            weights={"bce": 0.2, "dice": 0.4, "focal": 0.4},
            factory=lambda **kw: make_combined_loss(w_bce=0.2, w_dice=0.4, w_focal=0.4, symmetric_dice=True, **kw),
        ),
        "thebe_stable_combined": LossProfile(
            name="thebe_stable_combined",
            stage="thebe_finetune",
            description="Stability-focused BCE + Dice + Focal profile, weights 0.4/0.4/0.6.",
            weights={"bce": 0.4, "dice": 0.4, "focal": 0.6},
            factory=lambda **kw: make_combined_loss(w_bce=0.4, w_dice=0.4, w_focal=0.6, symmetric_dice=False, **kw),
        ),
        "swinunetr_monai_dice": LossProfile(
            name="swinunetr_monai_dice",
            stage="thebe_finetune",
            description="MONAI DiceLoss profile for SwinUNETR baseline segmentation.",
            weights={"dice": 1.0},
            factory=make_monai_dice_loss,
        ),
        "simmim_l1": LossProfile(
            name="simmim_l1",
            stage="simmim_pretrain",
            description="L1 reconstruction loss for masked seismic reconstruction.",
            weights={"l1": 1.0},
            factory=make_l1_loss,
        ),
        "sr_l1_vgg_gan": LossProfile(
            name="sr_l1_vgg_gan",
            stage="sr_training",
            description="SR composite: L1 + slice VGG + optional adversarial BCE.",
            weights={"l1": 1.0, "vgg": 0.05, "gan": 0.001},
            factory=make_sr_composite_loss,
        ),
        "segmentation_bce": LossProfile(
            name="segmentation_bce",
            stage="validation",
            description="Plain BCEWithLogits profile for validation-only pipelines.",
            weights={"bce": 1.0},
            factory=make_bce_loss,
        ),
    }
    return profiles.get(name, LossProfile(name=name, stage="unknown", description="", weights={}, factory=lambda **kw: None))


def build_loss(name: str, **kwargs: Any) -> Any:
    """Instantiate a loss module from the global LOSS_REGISTRY."""
    return LOSS_REGISTRY.get(name)(**kwargs)


@LOSS_REGISTRY.register("faultseg3d_combined_sym")
@LOSS_REGISTRY.register("thebe_combined_sym")
def make_combined_loss(
    *,
    w_bce: float = 0.2,
    w_dice: float = 0.4,
    w_focal: float = 0.4,
    symmetric_dice: bool = True,
    clamp_target: bool = True,
    pos_weight: float | None = None,
) -> Any:
    """Create the BCE + Dice + Focal segmentation loss used by 3D models."""

    torch, nn, _ = _require_torch()
    Dice = _binary_dice_loss_class(symmetric=symmetric_dice)
    Focal = _binary_focal_loss_class()

    class CombinedLoss(nn.Module):
        """Weighted BCE, Dice and focal segmentation loss."""

        def __init__(self) -> None:
            super().__init__()
            weight = torch.tensor([pos_weight]) if pos_weight is not None else None
            self.bce = nn.BCEWithLogitsLoss(pos_weight=weight)
            self.dice = Dice()
            self.focal = Focal()

        def forward(self, logits: Any, target: Any) -> Any:
            """Compute the combined segmentation loss."""

            if target.dim() == logits.dim() - 1:
                target = target.unsqueeze(1)
            target = target.to(dtype=logits.dtype, device=logits.device)
            if clamp_target:
                target = target.clamp(0.0, 1.0)
            return w_bce * self.bce(logits, target) + w_dice * self.dice(logits, target) + w_focal * self.focal(
                logits, target
            )

    return CombinedLoss()


@LOSS_REGISTRY.register("swinunetr_monai_dice")
def make_monai_dice_loss(**kwargs: Any) -> Any:
    """Create a MONAI ``DiceLoss`` with sigmoid activation enabled."""

    try:
        from monai.losses import DiceLoss
    except ImportError as exc:  # pragma: no cover
        raise ImportError("MONAI is required for the monai_dice loss profile") from exc
    return DiceLoss(sigmoid=True, include_background=False, **kwargs)


@LOSS_REGISTRY.register("simmim_l1")
def make_l1_loss(**kwargs: Any) -> Any:
    """Create a PyTorch ``L1Loss`` module."""

    _, nn, _ = _require_torch()
    return nn.L1Loss(**kwargs)


@LOSS_REGISTRY.register("segmentation_bce")
def make_bce_loss(**kwargs: Any) -> Any:
    """Create a PyTorch ``BCEWithLogitsLoss`` module."""

    _, nn, _ = _require_torch()
    return nn.BCEWithLogitsLoss(**kwargs)


@LOSS_REGISTRY.register("sr_l1_vgg_gan")
def make_sr_composite_loss(
    w_l1: float = 1.0,
    w_vgg: float = 0.05,
    w_adv: float = 0.001,
    use_vgg: bool = False,
) -> Any:
    """Create the SR composite loss with L1, optional VGG and GAN terms."""

    torch, nn, _ = _require_torch()

    class VGGLoss3D(nn.Module):
        """Slice-based VGG loss for 3D seismic volumes."""

        def __init__(self) -> None:
            super().__init__()
            if not use_vgg:
                self.vgg = None
                self.mse = nn.MSELoss()
                return
            try:
                from torchvision.models import VGG16_Weights, vgg16
            except ImportError as exc:  # pragma: no cover
                raise ImportError("torchvision is required for VGGLoss3D") from exc
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
            self.mse = nn.MSELoss()

        def forward(self, sr_cube: Any, hr_cube: Any) -> Any:
            """Compute slice-perceptual loss for one SR/HR volume batch."""

            if self.vgg is None:
                return self.mse(sr_cube, hr_cube)
            _, _, depth, height, width = sr_cube.shape
            losses = []
            slice_pairs = (
                (sr_cube[:, :, depth // 2, :, :], hr_cube[:, :, depth // 2, :, :]),
                (sr_cube[:, :, :, height // 2, :], hr_cube[:, :, :, height // 2, :]),
                (sr_cube[:, :, :, :, width // 2], hr_cube[:, :, :, :, width // 2]),
            )
            for sr_slice, hr_slice in slice_pairs:
                sr_norm = torch.clamp((sr_slice.float() + 3.0) / 6.0, 0.0, 1.0).repeat(1, 3, 1, 1)
                hr_norm = torch.clamp((hr_slice.float() + 3.0) / 6.0, 0.0, 1.0).repeat(1, 3, 1, 1)
                losses.append(self.mse(self.vgg(sr_norm), self.vgg(hr_norm)))
            return torch.stack(losses).mean()

    class SRCompositeLoss(nn.Module):
        """Composite SR loss module with inspectable last components."""

        def __init__(self) -> None:
            super().__init__()
            self.l1 = nn.L1Loss()
            self.vgg = VGGLoss3D()
            self.adv = nn.BCEWithLogitsLoss()
            self.last_components: dict[str, Any] = {}

        def forward(self, sr_cube: Any, hr_cube: Any, disc_logits: Any | None = None) -> Any:
            """Compute SR reconstruction and optional adversarial generator loss."""

            loss_l1 = self.l1(sr_cube, hr_cube)
            loss_vgg = self.vgg(sr_cube, hr_cube)
            loss_gan = torch.zeros((), dtype=loss_l1.dtype, device=loss_l1.device)
            loss = w_l1 * loss_l1 + w_vgg * loss_vgg
            if disc_logits is not None:
                target = torch.ones_like(disc_logits)
                loss_gan = self.adv(disc_logits, target)
                loss = loss + w_adv * loss_gan
            self.last_components = {
                "loss_l1": loss_l1.detach(),
                "loss_vgg": loss_vgg.detach(),
                "loss_gan": loss_gan.detach(),
                "loss_g": loss.detach(),
            }
            return loss

    return SRCompositeLoss()


@LOSS_REGISTRY.register("thebe_stable_combined")
def make_thebe_stable_combined(**kwargs: Any) -> Any:
    """Create a stability-focused combined loss."""
    return make_combined_loss(w_bce=0.4, w_dice=0.4, w_focal=0.6, symmetric_dice=False, **kwargs)


def _require_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - DataSphere has torch.
        raise ImportError("This loss profile requires PyTorch") from exc
    return torch, nn, F


def _binary_focal_loss_class() -> type[Any]:
    torch, nn, _ = _require_torch()

    class BinaryFocalLoss(nn.Module):
        """Binary focal loss implementation returned by the factory."""

        def __init__(
            self,
            gamma: float = 2.0,
            alpha: float = 0.25,
            reduction: str = "mean",
            eps: float = 1e-6,
        ) -> None:
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.reduction = reduction
            self.eps = eps

        def forward(self, logits: Any, target: Any) -> Any:
            """Compute focal loss from logits and binary targets."""

            probs = torch.sigmoid(logits)
            if target.dim() == probs.dim() - 1:
                target = target.unsqueeze(1)
            target = target.to(dtype=probs.dtype, device=probs.device)
            pos_mask = target.eq(1.0)
            pt = torch.where(pos_mask, probs, 1.0 - probs).clamp(self.eps, 1.0 - self.eps)
            alpha_t = torch.where(pos_mask, self.alpha, 1.0 - self.alpha)
            loss = -alpha_t * (1.0 - pt).pow(self.gamma) * torch.log(pt)
            if self.reduction == "sum":
                return loss.sum()
            if self.reduction == "none":
                return loss
            return loss.mean()

    return BinaryFocalLoss


def _binary_dice_loss_class(symmetric: bool) -> type[Any]:
    torch, nn, _ = _require_torch()

    class BinaryDiceLoss(nn.Module):
        """Binary Dice loss implementation returned by the factory."""

        def __init__(self, smooth: float = 1e-5) -> None:
            super().__init__()
            self.smooth = smooth

        def forward(self, logits: Any, target: Any) -> Any:
            """Compute Dice loss from logits and binary targets."""

            probs = torch.sigmoid(logits)
            if target.dim() == probs.dim() - 1:
                target = target.unsqueeze(1)
            target = target.to(dtype=probs.dtype, device=probs.device)
            probs_flat = probs.reshape(probs.size(0), -1)
            target_flat = target.reshape(target.size(0), -1)
            inter_pos = (probs_flat * target_flat).sum(dim=1)
            dice_pos = (2.0 * inter_pos + self.smooth) / (
                probs_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
            )
            if not symmetric:
                return 1.0 - dice_pos.mean()
            probs_neg = 1.0 - probs_flat
            target_neg = 1.0 - target_flat
            inter_neg = (probs_neg * target_neg).sum(dim=1)
            dice_neg = (2.0 * inter_neg + self.smooth) / (
                probs_neg.sum(dim=1) + target_neg.sum(dim=1) + self.smooth
            )
            return 1.0 - (0.5 * (dice_pos + dice_neg)).mean()

    return BinaryDiceLoss


# Backward-compatible aliases for direct imports.
def BinaryFocalLoss(*args: Any, **kwargs: Any) -> Any:
    """Instantiate the binary focal loss compatibility wrapper."""

    return _binary_focal_loss_class()(*args, **kwargs)


def BinaryDiceLoss(*args: Any, **kwargs: Any) -> Any:
    """Instantiate the foreground Dice loss compatibility wrapper."""

    return _binary_dice_loss_class(symmetric=False)(*args, **kwargs)


def BinaryDiceLossSymmetric(*args: Any, **kwargs: Any) -> Any:
    """Instantiate the symmetric foreground/background Dice loss wrapper."""

    return _binary_dice_loss_class(symmetric=True)(*args, **kwargs)


def CombinedLoss(*args: Any, **kwargs: Any) -> Any:
    """Instantiate the combined segmentation loss compatibility wrapper."""

    return make_combined_loss(*args, **kwargs)
