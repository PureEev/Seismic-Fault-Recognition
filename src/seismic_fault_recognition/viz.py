"""Visualization helpers for qualitative validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def sample_boolean_points(mask: np.ndarray, max_points: int = 200_000, seed: int = 42) -> np.ndarray:
    """Sample coordinates of true voxels from a boolean mask."""

    points = np.argwhere(np.asarray(mask).astype(bool))
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def make_3d_comparison_figure(
    prediction: np.ndarray,
    target: np.ndarray,
    max_points: int = 200_000,
    title: str = "Prediction vs ground truth",
) -> Any:
    """Build a Plotly 3D point comparison for prediction and target masks."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotly is required for 3D visualization") from exc

    pred_points = sample_boolean_points(prediction, max_points=max_points)
    target_points = sample_boolean_points(target, max_points=max_points)
    fig = go.Figure()
    if len(target_points):
        fig.add_trace(
            go.Scatter3d(
                x=target_points[:, 2],
                y=target_points[:, 1],
                z=target_points[:, 0],
                mode="markers",
                marker={"size": 1.5, "color": "rgba(31,119,180,0.45)"},
                name="ground truth",
            )
        )
    if len(pred_points):
        fig.add_trace(
            go.Scatter3d(
                x=pred_points[:, 2],
                y=pred_points[:, 1],
                z=pred_points[:, 0],
                mode="markers",
                marker={"size": 1.5, "color": "rgba(214,39,40,0.45)"},
                name="prediction",
            )
        )
    fig.update_layout(title=title, scene={"aspectmode": "data"})
    return fig


def save_3d_comparison_html(
    prediction: np.ndarray,
    target: np.ndarray,
    outfile: str | Path,
    max_points: int = 200_000,
) -> Path:
    """Save a 3D comparison Plotly figure as an HTML file."""

    fig = make_3d_comparison_figure(prediction, target, max_points=max_points)
    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    return path
