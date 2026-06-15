"""Model factory helpers for registry-driven training and validation."""

from __future__ import annotations

from typing import Any
from seismic_fault_recognition.registry import MODEL_REGISTRY

# Trigger registration by importing model modules.
from . import swinunetr_variants  # noqa: F401
from . import omniseis  # noqa: F401
from . import faultformer  # noqa: F401


def build_model_by_name(model_variant: str, **kwargs: Any) -> Any:
    """Build a model from the global MODEL_REGISTRY."""
    return MODEL_REGISTRY.get(model_variant)(**kwargs)
