"""Optional SEG-Y geometry and regular-grid helpers.

The functions that read SEG-Y import ``segyio`` lazily so the base package and
tests can run without preprocessing-only dependencies installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
import math

import numpy as np


@dataclass(frozen=True)
class ScalarCandidate:
    """Scored coordinate multiplier candidate for SEG-Y headers."""

    multiplier: float
    median_dx: float
    median_dy: float
    score: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


@dataclass(frozen=True)
class GridGeometry:
    """Estimated regular-grid orientation and spacing metadata."""

    azimuth_deg: float
    inline_step: float
    crossline_step: float
    inline_extent: float
    crossline_extent: float
    sample_count: int

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


def apply_scalar(value: float | int | None, scalar: float | int | None) -> float | None:
    """Apply a SEG-Y coordinate scalar to a header value."""

    if value is None:
        return None
    if scalar in (None, 0):
        return float(value)
    scalar_value = float(scalar)
    if scalar_value < 0:
        return float(value) / abs(scalar_value)
    return float(value) * scalar_value


def estimate_step(values: Sequence[float], min_abs_diff: float = 0.01) -> float:
    """Estimate the median non-zero step between sorted unique coordinates."""

    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size < 2:
        return float("nan")
    diffs = np.diff(arr)
    diffs = diffs[np.abs(diffs) > float(min_abs_diff)]
    if diffs.size == 0:
        return float("nan")
    return float(np.median(diffs))


def detect_scalar_and_coords(
    segy_path: str | Path,
    candidates: Sequence[float] = (1, 0.1, 0.01, 0.001, 0.0001, 10, 100, 1000, 10000),
) -> tuple[float, list[ScalarCandidate]]:
    """Score coordinate multipliers using CDP_X/CDP_Y spacing."""

    segyio = _require_segyio()
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as handle:
        x_raw = np.asarray(handle.attributes(segyio.TraceField.CDP_X)).astype(np.float64)
        y_raw = np.asarray(handle.attributes(segyio.TraceField.CDP_Y)).astype(np.float64)

    scored: list[ScalarCandidate] = []
    for multiplier in candidates:
        x = x_raw * float(multiplier)
        y = y_raw * float(multiplier)
        ux = np.sort(np.unique(np.round(x, 6)))
        uy = np.sort(np.unique(np.round(y, 6)))
        med_dx = float(np.median(np.diff(ux))) if ux.size > 1 else float("nan")
        med_dy = float(np.median(np.diff(uy))) if uy.size > 1 else float("nan")
        score = 0.0
        if 0.01 <= med_dx <= 10000 and 0.01 <= med_dy <= 10000:
            score += 10.0
        if np.isfinite(med_dx):
            score -= abs(med_dx - round(med_dx))
        if np.isfinite(med_dy):
            score -= abs(med_dy - round(med_dy))
        scored.append(ScalarCandidate(float(multiplier), med_dx, med_dy, float(score)))
    scored.sort(key=lambda item: item.score, reverse=True)
    if not scored:
        raise ValueError("No scalar candidates provided")
    return scored[0].multiplier, scored


def estimate_grid_geometry(segy_path: str | Path, max_scan: int = 100000) -> GridGeometry:
    """Estimate survey orientation and bin steps from CDP coordinates."""

    segyio = _require_segyio()
    xs: list[float] = []
    ys: list[float] = []
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as handle:
        total_traces = len(handle.trace)
        step = max(1, total_traces // max(1, int(max_scan)))
        for trace_index in range(0, total_traces, step):
            header = handle.header[trace_index]
            scalar = header.get(segyio.TraceField.SourceGroupScalar)
            cx = apply_scalar(header.get(segyio.TraceField.CDP_X), scalar)
            cy = apply_scalar(header.get(segyio.TraceField.CDP_Y), scalar)
            if cx is not None and cy is not None:
                xs.append(cx)
                ys.append(cy)

    if len(xs) < 2:
        raise ValueError("Not enough CDP coordinates to estimate grid geometry")

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    centered = np.vstack((x_arr - x_mean, y_arr - y_mean))
    u, _, _ = np.linalg.svd(centered, full_matrices=False)
    direction = u[:, 0]
    azimuth_rad = float(np.arctan2(direction[1], direction[0]))
    cos_a = float(np.cos(-azimuth_rad))
    sin_a = float(np.sin(-azimuth_rad))
    x_rot = cos_a * (x_arr - x_mean) - sin_a * (y_arr - y_mean)
    y_rot = sin_a * (x_arr - x_mean) + cos_a * (y_arr - y_mean)

    return GridGeometry(
        azimuth_deg=float(np.degrees(azimuth_rad)),
        inline_step=estimate_step(x_rot),
        crossline_step=estimate_step(y_rot),
        inline_extent=float(np.max(x_rot) - np.min(x_rot)),
        crossline_extent=float(np.max(y_rot) - np.min(y_rot)),
        sample_count=len(xs),
    )


def build_regular_cube_memmap(
    traces_file: str | Path,
    coords_file: str | Path,
    nsamples: int,
    out_cube: str | Path,
    out_counts: str | Path,
    dx: float | None = None,
    dy: float | None = None,
    max_bytes: int = 200 * 1024**3,
    chunk_traces: int = 100000,
    t_range: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Bin trace memmap data into a regular ``(nx * ny, nt)`` memmap cube."""

    coords = np.load(coords_file)
    n_traces = int(coords.shape[0])
    traces = np.memmap(traces_file, dtype="float32", mode="r", shape=(n_traces, int(nsamples)))
    if t_range is None:
        t0, t1 = 0, int(nsamples)
    else:
        t0, t1 = int(t_range[0]), int(t_range[1])
    if t0 < 0 or t1 <= t0 or t1 > int(nsamples):
        raise ValueError("Invalid t_range")
    nt = t1 - t0

    x = coords[:, 0].astype(np.float64)
    y = coords[:, 1].astype(np.float64)
    dx = float(dx) if dx is not None else _median_unique_step(x)
    dy = float(dy) if dy is not None else _median_unique_step(y)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    nx = int(math.floor((xmax - xmin) / dx)) + 1
    ny = int(math.floor((ymax - ymin) / dy)) + 1
    est_bytes = nx * ny * nt * 4
    if est_bytes > int(max_bytes):
        scale = math.sqrt(est_bytes / int(max_bytes))
        dx *= scale
        dy *= scale
        nx = int(math.floor((xmax - xmin) / dx)) + 1
        ny = int(math.floor((ymax - ymin) / dy)) + 1
        est_bytes = nx * ny * nt * 4
    if est_bytes > int(max_bytes):
        raise MemoryError("Requested cube is larger than max_bytes")

    flat_n = nx * ny
    cube_flat = np.memmap(out_cube, dtype="float32", mode="w+", shape=(flat_n, nt))
    counts = np.memmap(out_counts, dtype="uint32", mode="w+", shape=(flat_n,))
    cube_flat[:] = 0.0
    counts[:] = 0
    chunk_traces = max(1, int(chunk_traces))

    for start in range(0, n_traces, chunk_traces):
        stop = min(n_traces, start + chunk_traces)
        xs = x[start:stop]
        ys = y[start:stop]
        block = np.asarray(traces[start:stop, t0:t1], dtype=np.float32)
        ix = np.clip(np.rint((xs - xmin) / dx).astype(int), 0, nx - 1)
        iy = np.clip(np.rint((ys - ymin) / dy).astype(int), 0, ny - 1)
        flat_idx = iy * nx + ix
        np.add.at(cube_flat, flat_idx, block)
        np.add.at(counts, flat_idx, 1)

    nonzero = counts > 0
    cube_flat[nonzero] = cube_flat[nonzero] / counts[nonzero, None]
    cube_flat.flush()
    counts.flush()

    return {
        "nx": nx,
        "ny": ny,
        "nt": nt,
        "dx": dx,
        "dy": dy,
        "xmin": xmin,
        "ymin": ymin,
        "estimated_bytes": est_bytes,
        "out_cube": str(out_cube),
        "out_counts": str(out_counts),
    }


def _median_unique_step(values: np.ndarray) -> float:
    unique = np.sort(np.unique(values))
    if unique.size <= 1:
        return 1.0
    step = float(np.median(np.diff(unique)))
    return step if step > 0 else 1.0


def _require_segyio() -> object:
    try:
        import segyio  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise ImportError("SEG-Y preprocessing helpers require segyio") from exc
    return segyio
