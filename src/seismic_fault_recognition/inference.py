"""Inference helpers for validation notebooks and production workloads."""

from __future__ import annotations

from typing import Any, Callable, Sequence
from pathlib import Path

import numpy as np
from .logger import get_logger

logger = get_logger("sfr.inference")

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    """Apply a NumPy sigmoid to logits."""

    return 1.0 / (1.0 + np.exp(-logits))


def threshold_prediction(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probability scores to a boolean mask."""

    return np.asarray(probabilities) >= threshold


def sliding_window_inference(
    volume: Any,
    predictor: Callable[[Any], Any],
    roi_size: Sequence[int] = (128, 128, 128),
    overlap: float = 0.5,
    device: str | Any = "cuda",
) -> Any:
    """Run MONAI sliding-window inference when available, else predict directly."""

    try:
        from monai.inferers import sliding_window_inference as monai_sliding_window
    except ImportError:
        if torch is not None and hasattr(volume, "to"):
            with torch.no_grad():
                return predictor(volume.to(device))
        return predictor(volume)

    if torch is not None and hasattr(volume, "to"):
        with torch.no_grad():
            return monai_sliding_window(
                inputs=volume.to(device),
                roi_size=tuple(int(x) for x in roi_size),
                sw_batch_size=1,
                predictor=predictor,
                overlap=float(overlap),
                mode="gaussian",
            )
    return monai_sliding_window(
        inputs=volume,
        roi_size=tuple(int(x) for x in roi_size),
        sw_batch_size=1,
        predictor=predictor,
        overlap=float(overlap),
        mode="gaussian",
    )


def prepare_volume_np(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Prepare one NumPy seismic volume for model inference."""

    arr = np.nan_to_num(volume.astype(np.float32, copy=False), copy=False)
    return (arr - float(arr.mean())) / (float(arr.std()) + eps)


def _open_source(path: Path) -> Any:
    if path.suffix == ".zarr":
        import zarr
        return zarr.open(str(path), mode="r")
    if path.suffix == ".npy":
        return np.load(str(path), mmap_mode="r")
    if path.suffix == ".npz":
        # npz is an archive. We return the first array found or the one named 'data'/'arr_0'
        archive = np.load(str(path), mmap_mode="r")
        keys = list(archive.keys())
        if not keys:
            raise ValueError(f"Empty NPZ archive: {path}")
        # Prioritize common default keys, else pick the first one
        for preferred in ("data", "arr_0", "image", "label", "volume"):
            if preferred in keys:
                return archive[preferred]
        return archive[keys[0]]
    raise ValueError(f"Unsupported format for chunked inference: {path.suffix}")

def chunked_volume_inference(
    input_path: str | Path,
    output_path: str | Path,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    chunk_size: Sequence[int] = (256, 256, 256),
    roi_size: Sequence[int] = (128, 128, 128),
    overlap: float = 0.25,
    device: str = "cuda",
    use_zarr: bool = True,
) -> Path:
    """Run inference on a large volume from disk to disk using chunking.

    This function is designed for volumes that exceed available RAM.

    Args:
        input_path: Path to input .npz, .npy or .zarr.
        output_path: Path to save the resulting .zarr or .npy.
        predictor: Callable that takes a 5D torch.Tensor and returns logits.
        chunk_size: Processing chunk size.
        roi_size: Sliding window ROI size.
        overlap: Overlap between patches in sliding window.
        device: Torch device.
        use_zarr: Whether to use Zarr for intermediate/output storage.

    Returns:
        Path to the saved results.
    """

    import torch

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Open source (lazy if possible)
    source = _open_source(input_path)
    shape = source.shape
    logger.info(f"Starting chunked inference on volume with shape {shape}")

    # Prepare output
    if use_zarr:
        import zarr
        out_store = zarr.open(str(output_path), mode="w", shape=shape, chunks=chunk_size, dtype=np.float32)
    else:
        # Fallback to memmap if not using zarr
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_store = np.memmap(output_path, dtype=np.float32, mode="w+", shape=shape)

    # Calculate chunks
    d, h, w = shape
    cd, ch, cw = chunk_size

    for i in range(0, d, cd):
        for j in range(0, h, ch):
            for k in range(0, w, cw):
                # Define slice with small padding to avoid artifacts if possible,
                # but here we rely on sliding_window_inference's own overlap
                d_end = min(i + cd, d)
                h_end = min(j + ch, h)
                w_end = min(k + cw, w)

                logger.info(f"Processing chunk: depth[{i}:{d_end}], height[{j}:{h_end}], width[{k}:{w_end}]")

                # Load chunk to RAM
                chunk = source[i:d_end, j:h_end, k:w_end]

                # Normalize chunk (ideally should use global stats, but local is often fine in seismic)
                chunk_norm = prepare_volume_np(chunk)

                # Convert to torch and add B,C dims
                chunk_tensor = torch.from_numpy(chunk_norm).unsqueeze(0).unsqueeze(0)

                # Run sliding window on this chunk
                with torch.no_grad():
                    logits = sliding_window_inference(
                        chunk_tensor,
                        predictor,
                        roi_size=roi_size,
                        overlap=overlap,
                        device=device
                    )

                # Sigmoid and save back to disk
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                out_store[i:d_end, j:h_end, k:w_end] = probs

    if not use_zarr:
        out_store.flush()

    logger.info(f"Chunked inference completed. Results saved to {output_path}")
    return output_path
