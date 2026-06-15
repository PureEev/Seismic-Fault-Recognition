"""FastAPI application for seismic fault recognition services."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
from pathlib import Path
import io
import functools

from ..registry import MODEL_REGISTRY
from ..recipes import RECIPES
from ..logger import get_logger
from ..inference import _open_source
from .schemas import ModelInfo, RecipeInfo, InferenceRequest, InferenceTaskResponse, SystemStatus
from .worker import celery_app, run_inference_job

logger = get_logger("sfr.server")

app = FastAPI(
    title="Seismic Fault Recognition API",
    description="R&D API for 3D seismic segmentation and super-resolution.",
    version="0.1.0",
)

# Smart loading: cache open file handles to avoid disk I/O on every slice request
@functools.lru_cache(maxsize=5)
def _get_cached_source(path: str):
    """Load and cache the array source."""
    full_path = Path(path)
    if not full_path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")
    logger.info(f"Opening volume source: {path}")
    return _open_source(full_path)


@app.get("/", response_model=SystemStatus)
async def get_status():
    """Return system health and summary stats."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "models_available": len(MODEL_REGISTRY.list()),
        "recipes_available": len(RECIPES),
    }


@app.get("/models")
async def list_models():
    """List all registered neural network architectures."""
    return [{"name": name} for name in MODEL_REGISTRY.list()]


@app.get("/recipes")
async def list_recipes():
    """List all registered experiment recipes."""
    return [recipe.as_dict() for recipe in RECIPES.values()]


@app.post("/inference/submit", response_model=InferenceTaskResponse)
async def submit_inference(request: InferenceRequest):
    """Submit a 3D volume for segmentation via Celery worker."""

    if request.model_variant not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model variant: {request.model_variant}")

    if not Path(request.input_path).exists():
        raise HTTPException(status_code=404, detail=f"Input file not found: {request.input_path}")

    # Dispatch to Celery
    task = run_inference_job.delay(request.dict())

    return {
        "job_id": task.id,
        "status": "submitted",
        "message": "Inference task dispatched to worker queue."
    }


@app.get("/inference/status/{job_id}")
async def get_inference_status(job_id: str):
    """Check the status of a background inference job."""
    task_result = celery_app.AsyncResult(job_id)
    return {
        "job_id": job_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None
    }


@app.get("/volume/slice/{path:path}/{axis}/{index}")
async def get_volume_slice(path: str, axis: str, index: int):
    """Extract a 2D slice from a 3D volume for visualization."""
    try:
        import matplotlib.pyplot as plt

        source = _get_cached_source(path)

        # Map axis names
        axis_map = {'d': 0, 'h': 1, 'w': 2, '0': 0, '1': 1, '2': 2}
        ax_idx = axis_map.get(axis.lower())
        if ax_idx is None:
            raise HTTPException(status_code=400, detail="Invalid axis. Use d, h, w or 0, 1, 2")

        # Basic bounds check
        if index < 0 or index >= source.shape[ax_idx]:
            raise HTTPException(status_code=400, detail=f"Index out of bounds for axis {axis}")

        # Extract slice
        if ax_idx == 0:
            slice_data = source[index, :, :]
        elif ax_idx == 1:
            slice_data = source[:, index, :]
        else:
            slice_data = source[:, :, index]

        # Render to PNG
        buf = io.BytesIO()
        plt.imsave(buf, slice_data, cmap='gray', format='png')
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Slicing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/volume/3d/{path:path}")
async def get_volume_3d_html(
    path: str,
    threshold: float = 0.5,
    max_points: int = 200_000,
    z_start: int = 0, z_end: int = 512,
    y_start: int = 0, y_end: int = 512,
    x_start: int = 0, x_end: int = 512
):
    """Generate and return interactive 3D HTML visualization for a mask volume crop."""
    try:
        from ..viz import make_3d_comparison_figure
        import numpy as np

        source = _get_cached_source(path)

        # Apply crop
        # Ensure bounds are within the actual array shape
        d, h, w = source.shape
        z1, z2 = max(0, z_start), min(d, z_end)
        y1, y2 = max(0, y_start), min(h, y_end)
        x1, x2 = max(0, x_start), min(w, x_end)

        # Load the cropped array
        array = np.asarray(source[z1:z2, y1:y2, x1:x2])

        num_elements = np.prod(array.shape)
        if num_elements > 512 * 512 * 512:
            raise HTTPException(status_code=400, detail=f"Cropped volume too large ({num_elements} voxels). Max allowed is 512^3.")

        if num_elements == 0:
            raise HTTPException(status_code=400, detail="Cropped region is empty. Check your coordinates.")

        # Treat as mask if it has continuous probabilities
        mask = array >= threshold


        # We reuse the existing viz code. We pass empty array for target to just render one mask.
        empty_target = np.zeros((1, 1, 1), dtype=bool)

        fig = make_3d_comparison_figure(
            prediction=mask,
            target=empty_target,
            max_points=max_points,
            title=f"3D Volume Render: {Path(path).name}"
        )

        # Get raw HTML string
        html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')

        return HTMLResponse(content=html_str)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing visualization dependency: {e}")
    except Exception as e:
        logger.error(f"3D Rendering failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
