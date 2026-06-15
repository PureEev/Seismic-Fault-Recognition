"""Swin Tiny facade used by package and notebook workflows."""

from __future__ import annotations

from typing import Any


def build_swinunetr_tiny(**kwargs: Any) -> Any:
    """Build the final checkpoint-compatible Swin Tiny model."""

    from .swinunetr_variants import build_swinunetr_tiny as factory

    return factory(**kwargs)


def modify_swinunetr_model(model: Any) -> Any:
    """Apply the final Swin Tiny architecture to a compatible MONAI model."""

    from .swinunetr_variants import finalize_swinunetr_tiny

    return finalize_swinunetr_tiny(model)
