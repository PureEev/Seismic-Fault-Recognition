"""Pydantic schemas for the Seismic Fault Recognition API."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class ModelInfo(BaseModel):
    name: str
    description: Optional[str] = None
    checkpoint_compatibility: Optional[str] = None


class RecipeInfo(BaseModel):
    name: str
    stage: str
    dataset: str
    model_variant: str
    loss_profile: str
    trainer_profile: str


class InferenceRequest(BaseModel):
    input_path: str = Field(..., description="Path to input seismic volume (.npz, .npy, .zarr)")
    model_variant: str = Field(..., description="Model variant name from registry")
    checkpoint_path: str = Field(..., description="Path to model weights (.pth)")
    chunk_size: List[int] = [256, 256, 256]
    roi_size: List[int] = [128, 128, 128]
    overlap: float = 0.25


class InferenceTaskResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


class SystemStatus(BaseModel):
    status: str = "ok"
    version: str
    models_available: int
    recipes_available: int
