"""Model factories and architectures.

Heavy frameworks are imported inside factory functions so basic package imports
work even outside the training image.
"""

__all__ = [
    "build_swinunetr_tiny",
    "build_model_by_name",
    "modify_swinunetr_model",
    "OmniSeisEncoder",
    "OmniSeisSegmentationModel",
    "OmniSeisSRGenerator",
    "PatchGANDiscriminator3D",
    "AttentionFaultFormerNet",
]


def build_swinunetr_tiny(*args, **kwargs):
    """Build the default tiny SwinUNETR without importing MONAI at package import time."""

    from .swinunetr import build_swinunetr_tiny as factory

    return factory(*args, **kwargs)


def build_model_by_name(*args, **kwargs):
    """Build any registered model by factory name."""

    from .factory import build_model_by_name as factory

    return factory(*args, **kwargs)


def modify_swinunetr_model(*args, **kwargs):
    """Apply the package SwinUNETR modifier through a lazy import."""

    from .swinunetr import modify_swinunetr_model as modifier

    return modifier(*args, **kwargs)


def __getattr__(name: str):
    if name in {"OmniSeisEncoder", "OmniSeisSegmentationModel", "OmniSeisSRGenerator", "PatchGANDiscriminator3D"}:
        from . import omniseis

        return getattr(omniseis, name)
    if name == "AttentionFaultFormerNet":
        from .faultformer import AttentionFaultFormerNet

        return AttentionFaultFormerNet
    raise AttributeError(name)
