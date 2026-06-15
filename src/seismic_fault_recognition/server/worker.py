"""Celery configuration and tasks for distributed seismic processing."""

from __future__ import annotations

import os
from celery import Celery
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None

from ..logger import get_logger
from ..registry import MODEL_REGISTRY
from ..models.factory import build_model_by_name
from ..training import load_checkpoint
from ..inference import chunked_volume_inference

logger = get_logger("sfr.worker")

# Configure Celery
# Default to local redis if not specified via environment
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6373/0")

celery_app = Celery(
    "seismic_fault_recognition",
    broker=redis_url,
    backend=redis_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour limit for heavy 3D jobs
)

@celery_app.task(bind=True, name="run_inference_job")
def run_inference_job(self, request_data: dict):
    """Distributed task for chunked 3D volume inference."""

    job_id = self.request.id
    input_path = Path(request_data["input_path"])
    model_variant = request_data["model_variant"]
    checkpoint_path = request_data["checkpoint_path"]

    logger.info(f"Worker starting job {job_id} for {input_path}")

    try:
        if torch is None:
            raise ImportError("PyTorch is required for running inference jobs")

        # Update state to 'PROCESSING'
        self.update_state(state='PROGRESS', meta={'status': 'Loading model...'})

        # Build and load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model_by_name(model_variant).to(device)
        load_checkpoint(checkpoint_path, model=model, map_location=device)
        model.eval()

        def predictor(x):
            return model(x)

        output_path = input_path.with_suffix(f".{job_id}.zarr")

        # Define progress callback if needed, but chunked_volume_inference logs internally
        chunked_volume_inference(
            input_path=input_path,
            output_path=output_path,
            predictor=predictor,
            chunk_size=request_data.get("chunk_size", [256, 256, 256]),
            roi_size=request_data.get("roi_size", [128, 128, 128]),
            overlap=request_data.get("overlap", 0.25),
            device=device,
            use_zarr=True
        )

        logger.info(f"Job {job_id} completed. Output: {output_path}")
        return {
            "status": "success",
            "output_path": str(output_path),
            "job_id": job_id
        }

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        raise e
