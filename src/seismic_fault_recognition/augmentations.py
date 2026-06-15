"""Stage-specific augmentation and preprocessing profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
from seismic_fault_recognition.registry import AUGMENTATION_REGISTRY


@dataclass(frozen=True)
class AugmentationProfile:
    """Registered augmentation pipeline description.

    Attributes:
        name: Stable registry key.
        stage: Experiment stage that uses the profile.
        steps: Human-readable processing steps.
        builder: Callable that creates the runtime transform/pipeline object.
        description: Short documentation string for CLI and docs output.
    """

    name: str
    stage: str
    steps: tuple[str, ...]
    builder: Callable[..., Any]
    description: str


def list_augmentation_profiles() -> tuple[str, ...]:
    """Return registered augmentation profile names."""
    return tuple(AUGMENTATION_REGISTRY.list())


def build_augmentation(name: str, **kwargs: Any) -> Any:
    """Build a runtime augmentation object from the global AUGMENTATION_REGISTRY."""
    return AUGMENTATION_REGISTRY.get(name)(**kwargs)


def get_augmentation_profile(name: str) -> AugmentationProfile:
    """Return a registered augmentation profile by name (compatibility alias)."""
    profiles = {
        "faultseg3d_train": AugmentationProfile(
            name="faultseg3d_train",
            stage="faultseg3d_pretrain",
            steps=("EnsureChannelFirstd", "ScaleIntensityRanged", "RandAffined", "Rand3DElasticd", "RandGaussianNoised", "ToTensord"),
            builder=lambda **kw: build_faultseg3d_transforms(train=True),
            description="Synthetic FaultSeg3D supervised augmentation profile.",
        ),
        "faultseg3d_val": AugmentationProfile(
            name="faultseg3d_val",
            stage="faultseg3d_pretrain",
            steps=("EnsureChannelFirstd", "ScaleIntensityRanged", "ToTensord"),
            builder=lambda **kw: build_faultseg3d_transforms(train=False),
            description="FaultSeg3D validation transform profile.",
        ),
        "thebe_random_crop": AugmentationProfile(
            name="thebe_random_crop",
            stage="thebe_finetune",
            steps=("pad_to_min_shape", "shared_random_crop", "zscore_seismic", "binary_fault_mask"),
            builder=build_thebe_crop_pipeline,
            description="Thebe paired crop and z-score profile used by fine-tuning notebooks.",
        ),
        "simmim_masking": AugmentationProfile(
            name="simmim_masking",
            stage="simmim_pretrain",
            steps=("random_grid_mask", "kron_upsample_mask", "mask_token_fill"),
            builder=build_simmim_masking,
            description="3D SimMIM masking profile.",
        ),
        "sr_degradation": AugmentationProfile(
            name="sr_degradation",
            stage="sr_training",
            steps=("gaussian_blur", "optional_trace_drop", "snr_noise", "zscore_lr_hr"),
            builder=build_sr_degradation_pipeline,
            description="On-the-fly low-resolution degradation profile for SR training.",
        ),
    }
    return profiles.get(name, AugmentationProfile(name=name, stage="unknown", steps=(), builder=lambda **kw: None, description=""))


@AUGMENTATION_REGISTRY.register("faultseg3d_train")
def build_faultseg3d_train(**kwargs: Any) -> Any:
    """Build the FaultSeg3D training transform pipeline."""
    return build_faultseg3d_transforms(train=True)


@AUGMENTATION_REGISTRY.register("faultseg3d_val")
def build_faultseg3d_val(**kwargs: Any) -> Any:
    """Build the FaultSeg3D validation transform pipeline."""
    return build_faultseg3d_transforms(train=False)


def build_faultseg3d_transforms(train: bool = True) -> Any:
    """MONAI transforms for FaultSeg3D pretraining."""

    try:
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            Rand3DElasticd,
            RandAffined,
            RandGaussianNoised,
            ScaleIntensityRanged,
            ToTensord,
        )
    except ImportError:
        return {
            "type": "monai_unavailable",
            "train": train,
        }

    base = [
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        ScaleIntensityRanged(keys=["image"], a_min=-1.0, a_max=1.0, b_min=-1.0, b_max=1.0, clip=True),
    ]
    if train:
        base.extend(
            [
                RandAffined(
                    keys=["image", "label"],
                    mode=("bilinear", "nearest"),
                    prob=0.5,
                    rotate_range=(0.0, 0.0, 0.15),
                    scale_range=(0.05, 0.05, 0.05),
                    padding_mode="border",
                ),
                Rand3DElasticd(
                    keys=["image", "label"],
                    sigma_range=(3.0, 7.0),
                    magnitude_range=(5.0, 30.0),
                    prob=0.3,
                    mode=("bilinear", "nearest"),
                    padding_mode="border",
                ),
                RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
            ]
        )
    base.append(ToTensord(keys=["image", "label"]))
    return Compose(base)


@AUGMENTATION_REGISTRY.register("thebe_random_crop")
def build_thebe_crop_pipeline(
    target_shape: Sequence[int] = (128, 128, 128),
    random_crop: bool = True,
    normalize: bool = True,
) -> dict[str, Any]:
    """Thebe uses dataset-level crop/clean logic rather than MONAI transforms."""

    return {
        "type": "dataset_crop",
        "target_shape": tuple(int(x) for x in target_shape),
        "random_crop": bool(random_crop),
        "normalize": bool(normalize),
    }


@AUGMENTATION_REGISTRY.register("simmim_masking")
def build_simmim_masking(
    input_size: Sequence[int] = (128, 128, 128),
    mask_patch_size: int = 16,
    mask_ratio: float = 0.6,
    seed: int | None = None,
) -> Any:
    """Create the 3D SimMIM mask generator used by reconstruction pretraining."""

    from .data import MaskGenerator3D

    return MaskGenerator3D(
        input_size=input_size,
        mask_patch_size=mask_patch_size,
        mask_ratio=mask_ratio,
        seed=seed,
    )


@AUGMENTATION_REGISTRY.register("sr_degradation")
def build_sr_degradation_pipeline(
    snr_range: tuple[float, float] = (10.0, 20.0),
    z_sigma_range: tuple[float, float] = (1.5, 2.5),
    xy_sigma_range: tuple[float, float] = (0.5, 1.2),
    is_train: bool = True,
    depth_jitter_prob: float = 0.30,
    scale_mask_prob: float = 0.30,
    trace_drop_prob: float = 0.30,
    trace_drop_ratio: tuple[float, float] = (0.01, 0.02),
    seed: int | None = None,
) -> Callable[[Any], Any]:
    """On-the-fly low-resolution degradation for SR training."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("SR degradation requires numpy") from exc

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        gaussian_filter = None

    rng = np.random.default_rng(seed)

    def degrade(cube: Any) -> Any:
        """Return one degraded low-resolution cube."""

        arr = np.asarray(cube, dtype=np.float32)
        if gaussian_filter is not None:
            z_sig = rng.uniform(*z_sigma_range)
            xy_sig = rng.uniform(*xy_sigma_range)
            lr_cube = gaussian_filter(arr, sigma=(z_sig, xy_sig, xy_sig))
        else:
            lr_cube = arr.copy()
        if is_train:
            if rng.random() < depth_jitter_prob:
                shifts = rng.integers(-1, 2, size=(arr.shape[1], arr.shape[2]))
                z_ind = np.arange(arr.shape[0])[:, None, None]
                y_ind = np.arange(arr.shape[1])[None, :, None]
                x_ind = np.arange(arr.shape[2])[None, None, :]
                lr_cube = lr_cube[np.clip(z_ind - shifts, 0, arr.shape[0] - 1), y_ind, x_ind]
            if rng.random() < scale_mask_prob:
                scale_mask = rng.uniform(0.6, 1.4, size=(1, arr.shape[1], arr.shape[2]))
                if gaussian_filter is not None:
                    scale_mask = gaussian_filter(scale_mask, sigma=(0, 2, 2))
                lr_cube = lr_cube * scale_mask
            if rng.random() < trace_drop_prob:
                drop_ratio = rng.uniform(*trace_drop_ratio)
                trace_mask = rng.choice([0.0, 1.0], size=(1, arr.shape[1], arr.shape[2]), p=[drop_ratio, 1 - drop_ratio])
                lr_cube = lr_cube * trace_mask
        snr_db = rng.uniform(*snr_range)
        noise_power = float(np.mean(lr_cube**2)) / (10 ** (snr_db / 10.0) + 1e-8)
        noise = rng.normal(0.0, np.sqrt(noise_power), size=lr_cube.shape).astype(np.float32)
        return lr_cube + noise

    return degrade
