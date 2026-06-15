"""2D SeisGAN preprocessing utilities.

This module extracts inline/crossline 2D patches from a 3D cube and creates
LR/HR pairs through simple seismic degradations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class Slice2DRecord:
    """A 2D patch plus its location in the source cube."""

    orientation: str
    slice_index: int
    trace_start: int
    depth_start: int
    array: np.ndarray


def extract_slice_records(
    cube: np.ndarray,
    patch: Sequence[int] = (256, 256),
    inline_axis: int = 0,
    crossline_axis: int = 1,
    depth_axis: int = 2,
    slice_stride: int = 16,
    stride: int = 128,
) -> list[Slice2DRecord]:
    """Extract inline and crossline 2D patches from a 3D cube.

    Axes are normalized to ``inline, crossline, depth`` before extraction.
    Patches smaller than ``patch`` are skipped.
    """

    patch_traces, patch_depth = _pair(patch, "patch")
    slice_stride = _positive_int(slice_stride, "slice_stride")
    stride = _positive_int(stride, "stride")
    normalized = np.moveaxis(np.asarray(cube), (inline_axis, crossline_axis, depth_axis), (0, 1, 2))
    inline_len, crossline_len, depth_len = normalized.shape
    records: list[Slice2DRecord] = []

    if crossline_len >= patch_traces and depth_len >= patch_depth:
        for inline in range(0, inline_len, slice_stride):
            section = normalized[inline, :, :]
            records.extend(
                _window_records(
                    section,
                    orientation="inline",
                    slice_index=inline,
                    patch=(patch_traces, patch_depth),
                    stride=stride,
                )
            )

    if inline_len >= patch_traces and depth_len >= patch_depth:
        for crossline in range(0, crossline_len, slice_stride):
            section = normalized[:, crossline, :]
            records.extend(
                _window_records(
                    section,
                    orientation="crossline",
                    slice_index=crossline,
                    patch=(patch_traces, patch_depth),
                    stride=stride,
                )
            )

    return records


def extract_inline_crossline_slices(
    cube: np.ndarray,
    patch: Sequence[int] = (256, 256),
    inline_axis: int = 0,
    crossline_axis: int = 1,
    depth_axis: int = 2,
    slice_stride: int = 16,
    stride: int = 128,
) -> list[np.ndarray]:
    """Return only extracted patch arrays for lightweight callers."""

    return [
        record.array
        for record in extract_slice_records(
            cube,
            patch=patch,
            inline_axis=inline_axis,
            crossline_axis=crossline_axis,
            depth_axis=depth_axis,
            slice_stride=slice_stride,
            stride=stride,
        )
    ]


def is_empty_slice(slice2d: np.ndarray, zero_threshold: float = 0.7, variance_threshold: float = 1e-6) -> bool:
    """Return whether a 2D slice is mostly zero or nearly constant."""

    arr = np.asarray(slice2d)
    zero_ratio = float((arr == 0).sum()) / max(arr.size, 1)
    return zero_ratio > float(zero_threshold) or float(np.var(arr)) < float(variance_threshold)


def filter_empty_slices(
    slices: Iterable[np.ndarray],
    zero_threshold: float = 0.7,
    variance_threshold: float = 1e-6,
) -> list[np.ndarray]:
    """Filter out empty or near-constant 2D slices."""

    return [
        np.asarray(item)
        for item in slices
        if not is_empty_slice(item, zero_threshold=zero_threshold, variance_threshold=variance_threshold)
    ]


def normalize_minmax(slice2d: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Normalize a 2D slice into ``[0, 1]`` using per-slice min/max."""

    arr = np.asarray(slice2d, dtype=np.float32)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32, copy=False)


def ricker_wavelet(f0: float, dt: float, length_s: float = 0.128) -> np.ndarray:
    """Create a normalized Ricker wavelet.

    Args:
        f0: Dominant frequency in Hz.
        dt: Sampling interval in seconds.
        length_s: Wavelet length in seconds.

    Returns:
        One-dimensional float32 wavelet.
    """

    nt = max(3, int(round(float(length_s) / float(dt))))
    time = np.linspace(-float(length_s) / 2.0, float(length_s) / 2.0, nt, dtype=np.float32)
    pi2 = (np.pi * float(f0)) ** 2
    wavelet = (1.0 - 2.0 * pi2 * time**2) * np.exp(-pi2 * time**2)
    peak = float(np.max(np.abs(wavelet)))
    if peak == 0.0:
        return wavelet.astype(np.float32, copy=False)
    return (wavelet / peak).astype(np.float32, copy=False)


def lowpass_time(slice2d: np.ndarray, f0: float, dt: float = 0.002, length_s: float = 0.128) -> np.ndarray:
    """Apply a Ricker low-pass convolution along the depth/time axis."""

    arr = np.asarray(slice2d, dtype=np.float32)
    wavelet = ricker_wavelet(f0=f0, dt=dt, length_s=length_s)
    out = np.empty_like(arr, dtype=np.float32)
    for trace_idx in range(arr.shape[0]):
        out[trace_idx, :] = _convolve_same_length(arr[trace_idx, :], wavelet)
    return out


def add_noise_by_snr(
    slice2d: np.ndarray,
    snr_db: float,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise with the requested signal-to-noise ratio."""

    arr = np.asarray(slice2d, dtype=np.float32)
    power = float(np.mean(arr**2))
    if power == 0.0:
        return arr.copy()
    generator = rng or np.random.default_rng(seed)
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    sigma = np.sqrt(power / snr_linear)
    noise = generator.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    return arr + noise


def tsp_downsample(slice2d: np.ndarray, down_space: bool = True, down_time: bool = True) -> np.ndarray:
    """Downsample a 2D seismic patch by taking every second trace/sample."""

    s_idx = slice(None, None, 2) if down_space else slice(None)
    t_idx = slice(None, None, 2) if down_time else slice(None)
    return np.asarray(slice2d)[s_idx, t_idx]


def make_lr_hr_pairs(
    slices: Iterable[np.ndarray],
    apply_degradations: bool = True,
    f0: float = 15.0,
    snr_db: float = 10.0,
    down_space: bool = True,
    down_time: bool = True,
    seed: int | None = None,
    zero_threshold: float = 0.7,
    variance_threshold: float = 1e-6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create normalized LR/HR pairs from extracted 2D slices."""

    rng = np.random.default_rng(seed)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for hr_slice in filter_empty_slices(
        slices,
        zero_threshold=zero_threshold,
        variance_threshold=variance_threshold,
    ):
        if apply_degradations:
            lr_slice = lowpass_time(hr_slice, f0=f0)
            lr_slice = add_noise_by_snr(lr_slice, snr_db=snr_db, rng=rng)
            lr_slice = tsp_downsample(lr_slice, down_space=down_space, down_time=down_time)
        else:
            lr_slice = np.asarray(hr_slice).copy()
        pairs.append((normalize_minmax(lr_slice), normalize_minmax(hr_slice)))
    return pairs


def save_h5(array: np.ndarray, path: str | Path) -> None:
    """Save one array into an HDF5 file under the ``/data`` dataset."""

    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise ImportError("Saving SeisGAN pairs to HDF5 requires h5py") from exc

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        handle.create_dataset("/data", data=np.asarray(array))


def save_lr_hr_pairs_h5(base_dir: str | Path, pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
    """Save LR/HR pairs to ``base_dir/low`` and ``base_dir/high`` HDF5 files."""

    base = Path(base_dir)
    low_dir = base / "low"
    high_dir = base / "high"
    low_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)
    for index, (lr_slice, hr_slice) in enumerate(pairs):
        save_h5(lr_slice, low_dir / f"case_{index}.h5")
        save_h5(hr_slice, high_dir / f"case_{index}.h5")


class DatasetConverterSeisGAN:
    """Convenience wrapper around the SeisGAN preprocessing helpers."""

    def __init__(self, files_path: str | Path | None = None) -> None:
        self.files_path = files_path

    def cut_cube_to_slices(
        self,
        cube: np.ndarray,
        patch: Sequence[int] = (256, 256),
        inline_index: int = 0,
        crossline_index: int = 1,
        depth_index: int = 2,
        slice_stride: int = 16,
        stride: int = 128,
    ) -> list[np.ndarray]:
        """Extract inline and crossline patch arrays from a 3D cube."""

        return extract_inline_crossline_slices(
            cube,
            patch=patch,
            inline_axis=inline_index,
            crossline_axis=crossline_index,
            depth_axis=depth_index,
            slice_stride=slice_stride,
            stride=stride,
        )

    def is_empty(self, slice2d: np.ndarray, zero_thr: float = 0.7, var_thr: float = 1e-6) -> bool:
        """Return whether a 2D slice should be discarded as empty."""

        return is_empty_slice(slice2d, zero_threshold=zero_thr, variance_threshold=var_thr)

    def clear_imgs(self, slices: Iterable[np.ndarray]) -> list[np.ndarray]:
        """Filter empty slices from an iterable of 2D arrays."""

        return filter_empty_slices(slices)

    def normalize_slice(self, slice2d: np.ndarray) -> np.ndarray:
        """Min/max normalize one 2D slice."""

        return normalize_minmax(slice2d)

    def ricker_wavelet(self, f0: float, dt: float, length_s: float = 0.128) -> np.ndarray:
        """Create a normalized Ricker wavelet."""

        return ricker_wavelet(f0=f0, dt=dt, length_s=length_s)

    def lowpass_time(self, slice2d: np.ndarray, f0: float, dt: float = 0.002, length_s: float = 0.128) -> np.ndarray:
        """Apply low-pass Ricker convolution along the time axis."""

        return lowpass_time(slice2d, f0=f0, dt=dt, length_s=length_s)

    def add_noise_by_snr(self, slice2d: np.ndarray, snr_db: float) -> np.ndarray:
        """Add Gaussian noise at the requested signal-to-noise ratio."""

        return add_noise_by_snr(slice2d, snr_db=snr_db)

    def tsp_downsample(self, slice2d: np.ndarray, down_space: bool = True, down_time: bool = True) -> np.ndarray:
        """Downsample a 2D slice in trace and/or time directions."""

        return tsp_downsample(slice2d, down_space=down_space, down_time=down_time)

    def process(
        self,
        slices: Iterable[np.ndarray],
        apply_degradations: bool = True,
        f0: float = 15.0,
        snr_db: float = 10.0,
        down_space: bool = True,
        down_time: bool = True,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Create LR/HR pairs from extracted slices."""

        return make_lr_hr_pairs(
            slices,
            apply_degradations=apply_degradations,
            f0=f0,
            snr_db=snr_db,
            down_space=down_space,
            down_time=down_time,
        )

    def save_files(self, base_dir: str | Path, pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
        """Save LR/HR pairs as HDF5 files under ``base_dir``."""

        save_lr_hr_pairs_h5(base_dir, pairs)


def _window_records(
    section: np.ndarray,
    orientation: str,
    slice_index: int,
    patch: tuple[int, int],
    stride: int,
) -> list[Slice2DRecord]:
    patch_traces, patch_depth = patch
    windows = sliding_window_view(section, window_shape=patch)
    records: list[Slice2DRecord] = []
    for trace_start in range(0, section.shape[0] - patch_traces + 1, stride):
        for depth_start in range(0, section.shape[1] - patch_depth + 1, stride):
            patch_img = windows[trace_start, depth_start].copy()
            records.append(
                Slice2DRecord(
                    orientation=orientation,
                    slice_index=int(slice_index),
                    trace_start=int(trace_start),
                    depth_start=int(depth_start),
                    array=patch_img,
                )
            )
    return records


def _convolve_same_length(trace: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    full_same = np.convolve(trace, kernel, mode="same")
    if full_same.shape[0] == trace.shape[0]:
        return full_same.astype(np.float32, copy=False)
    start = max(0, (full_same.shape[0] - trace.shape[0]) // 2)
    return full_same[start : start + trace.shape[0]].astype(np.float32, copy=False)


def _pair(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(tuple(value)) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    first, second = tuple(int(x) for x in value)
    if first <= 0 or second <= 0:
        raise ValueError(f"{name} values must be positive")
    return first, second


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed
