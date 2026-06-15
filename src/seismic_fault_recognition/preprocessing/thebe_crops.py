"""Memmap-safe Thebe cube assembly and crop extraction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VolumeBounds:
    """Inclusive/exclusive valid bounds for a Thebe 3D volume."""

    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int

    def as_slices(self) -> tuple[slice, slice, slice]:
        """Return bounds as NumPy slices."""

        return (
            slice(self.x_min, self.x_max),
            slice(self.y_min, self.y_max),
            slice(self.z_min, self.z_max),
        )

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


@dataclass(frozen=True)
class CropRecord:
    """Metadata for one saved paired Thebe crop."""

    filename: str
    x: int
    y: int
    z: int
    seis_path: str
    fault_path: str


@dataclass(frozen=True)
class ExtractionReport:
    """Summary returned after saving valid crop files."""

    bounds: VolumeBounds | None
    output_dir: str
    saved_count: int
    skipped_footprint_count: int
    skipped_empty_count: int
    records: tuple[CropRecord, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""

        return {
            "bounds": self.bounds.as_dict() if self.bounds is not None else None,
            "output_dir": self.output_dir,
            "saved_count": self.saved_count,
            "skipped_footprint_count": self.skipped_footprint_count,
            "skipped_empty_count": self.skipped_empty_count,
            "records": [asdict(record) for record in self.records],
        }


def assemble_npz_chunks_to_memmap(
    directory: str | Path,
    prefix: str,
    out_filepath: str | Path,
    num_chunks: int = 9,
    axis: int = 0,
    key: str | None = None,
) -> np.memmap:
    """Concatenate ``prefix1.npz`` ... ``prefixN.npz`` into one ``.npy`` memmap."""

    source_dir = Path(directory)
    out_path = Path(out_filepath)
    if out_path.exists():
        return np.lib.format.open_memmap(out_path, mode="r")

    chunks: list[Path] = [source_dir / f"{prefix}{index}.npz" for index in range(1, int(num_chunks) + 1)]
    if not chunks:
        raise ValueError("num_chunks must be positive")
    for chunk_path in chunks:
        if not chunk_path.exists():
            raise FileNotFoundError(chunk_path)

    first_key, first_array = _load_npz_array(chunks[0], key)
    array_key = key or first_key
    shape = list(first_array.shape)
    dtype = first_array.dtype
    total_axis_len = shape[axis]
    for chunk_path in chunks[1:]:
        _, chunk = _load_npz_array(chunk_path, array_key)
        total_axis_len += int(chunk.shape[axis])
    final_shape = tuple(shape[:axis] + [total_axis_len] + shape[axis + 1 :])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=final_shape)
    current = 0
    for chunk_path in chunks:
        _, chunk = _load_npz_array(chunk_path, array_key)
        chunk_len = int(chunk.shape[axis])
        target = [slice(None)] * len(final_shape)
        target[axis] = slice(current, current + chunk_len)
        mmap[tuple(target)] = chunk
        current += chunk_len
    mmap.flush()
    return np.lib.format.open_memmap(out_path, mode="r")


def find_valid_bounds(
    volume: np.ndarray,
    pad_value: float | int | None = 0.0,
    chunk_size: int = 50,
) -> tuple[VolumeBounds | None, np.ndarray]:
    """Find valid data bounds and a 2D spatial footprint without loading all chunks."""

    shape = tuple(int(x) for x in volume.shape)
    if len(shape) != 3:
        raise ValueError("volume must be a 3D array")
    chunk_size = max(1, int(chunk_size))
    x_proj = np.zeros(shape[0], dtype=bool)
    y_proj = np.zeros(shape[1], dtype=bool)
    z_proj = np.zeros(shape[2], dtype=bool)
    footprint_2d = np.zeros((shape[0], shape[1]), dtype=bool)

    for x_start in range(0, shape[0], chunk_size):
        chunk = np.asarray(volume[x_start : x_start + chunk_size])
        valid_mask = ~np.isnan(chunk)
        if pad_value is not None:
            valid_mask &= chunk != pad_value
        x_stop = x_start + chunk.shape[0]
        x_proj[x_start:x_stop] = np.any(valid_mask, axis=(1, 2))
        y_proj |= np.any(valid_mask, axis=(0, 2))
        z_proj |= np.any(valid_mask, axis=(0, 1))
        footprint_2d[x_start:x_stop, :] = np.any(valid_mask, axis=2)

    x_idx = np.where(x_proj)[0]
    y_idx = np.where(y_proj)[0]
    z_idx = np.where(z_proj)[0]
    if len(x_idx) == 0 or len(y_idx) == 0 or len(z_idx) == 0:
        return None, footprint_2d
    return (
        VolumeBounds(
            x_min=int(x_idx[0]),
            x_max=int(x_idx[-1] + 1),
            y_min=int(y_idx[0]),
            y_max=int(y_idx[-1] + 1),
            z_min=int(z_idx[0]),
            z_max=int(z_idx[-1] + 1),
        ),
        footprint_2d,
    )


def analyze_seismic_volume_memmap(
    volume_mmap: np.ndarray,
    pad_value: float | int | None = 0.0,
    chunk_size: int = 50,
) -> tuple[dict[str, int] | None, np.ndarray]:
    """Return valid bounds as a dictionary plus the spatial footprint."""

    bounds, footprint = find_valid_bounds(volume_mmap, pad_value=pad_value, chunk_size=chunk_size)
    return (bounds.as_dict() if bounds is not None else None), footprint


def crop_grid(
    shape: Sequence[int],
    crop_size: Sequence[int] = (128, 128, 128),
    overlap: float = 0.25,
) -> list[tuple[int, int, int]]:
    """Return crop origins with final edge coverage for each axis."""

    shape3 = _triple(shape, "shape")
    crop3 = _triple(crop_size, "crop_size")
    starts = [_axis_starts(dim, size, overlap) for dim, size in zip(shape3, crop3)]
    return [(x, y, z) for x in starts[0] for y in starts[1] for z in starts[2]]


def save_valid_crops(
    seis_volume: np.ndarray,
    fault_volume: np.ndarray,
    output_dir: str | Path,
    crop_size: Sequence[int] = (128, 128, 128),
    overlap: float = 0.25,
    prefix: str = "thebe",
    min_footprint: float = 0.90,
    min_std: float = 1e-6,
    pad_value: float | int | None = 0.0,
    chunk_size: int = 50,
    compressed: bool = True,
) -> ExtractionReport:
    """Save valid paired Thebe crops under ``output_dir/seis`` and ``output_dir/fault``."""

    crop3 = _triple(crop_size, "crop_size")
    out_dir = Path(output_dir)
    seis_dir = out_dir / "seis"
    fault_dir = out_dir / "fault"
    seis_dir.mkdir(parents=True, exist_ok=True)
    fault_dir.mkdir(parents=True, exist_ok=True)

    bounds, footprint = find_valid_bounds(seis_volume, pad_value=pad_value, chunk_size=chunk_size)
    if bounds is None:
        return ExtractionReport(None, str(out_dir), 0, 0, 0, ())

    seis_view = seis_volume[bounds.as_slices()]
    fault_view = fault_volume[bounds.as_slices()]
    footprint_view = footprint[bounds.x_min : bounds.x_max, bounds.y_min : bounds.y_max]
    records: list[CropRecord] = []
    skipped_footprint = 0
    skipped_empty = 0

    x_starts = _axis_starts(seis_view.shape[0], crop3[0], overlap)
    y_starts = _axis_starts(seis_view.shape[1], crop3[1], overlap)
    z_starts = _axis_starts(seis_view.shape[2], crop3[2], overlap)
    saver = np.savez_compressed if compressed else np.savez

    for x in x_starts:
        for y in y_starts:
            footprint_crop = footprint_view[x : x + crop3[0], y : y + crop3[1]]
            if float(np.mean(footprint_crop)) < float(min_footprint):
                skipped_footprint += len(z_starts)
                continue
            for z in z_starts:
                seis_crop = np.asarray(seis_view[x : x + crop3[0], y : y + crop3[1], z : z + crop3[2]])
                if float(np.std(seis_crop)) < float(min_std):
                    skipped_empty += 1
                    continue
                fault_crop = np.asarray(fault_view[x : x + crop3[0], y : y + crop3[1], z : z + crop3[2]])
                abs_x = bounds.x_min + x
                abs_y = bounds.y_min + y
                abs_z = bounds.z_min + z
                filename = f"{prefix}_x{abs_x}_y{abs_y}_z{abs_z}.npz"
                seis_path = seis_dir / filename
                fault_path = fault_dir / filename
                saver(seis_path, arr_0=seis_crop.astype(np.float32, copy=False))
                saver(fault_path, arr_0=fault_crop.astype(np.int8, copy=False))
                records.append(
                    CropRecord(
                        filename=filename,
                        x=int(abs_x),
                        y=int(abs_y),
                        z=int(abs_z),
                        seis_path=str(seis_path),
                        fault_path=str(fault_path),
                    )
                )

    return ExtractionReport(
        bounds=bounds,
        output_dir=str(out_dir),
        saved_count=len(records),
        skipped_footprint_count=skipped_footprint,
        skipped_empty_count=skipped_empty,
        records=tuple(records),
    )


def _load_npz_array(path: Path, key: str | None) -> tuple[str, np.ndarray]:
    with np.load(path) as data:
        array_key = key or data.files[0]
        if array_key not in data.files:
            raise KeyError(f"{array_key!r} not found in {path}")
        return array_key, np.asarray(data[array_key])


def _axis_starts(dim: int, crop: int, overlap: float) -> list[int]:
    dim = int(dim)
    crop = int(crop)
    if crop <= 0:
        raise ValueError("crop sizes must be positive")
    if dim < crop:
        return []
    stride = int(crop * (1.0 - float(overlap)))
    if stride <= 0:
        raise ValueError("overlap must be less than 1.0")
    starts = list(range(0, dim - crop + 1, stride))
    if starts and starts[-1] + crop < dim:
        starts.append(dim - crop)
    if not starts and dim == crop:
        starts.append(0)
    return starts


def _triple(value: Sequence[int], name: str) -> tuple[int, int, int]:
    parsed = tuple(int(x) for x in value)
    if len(parsed) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if any(item <= 0 for item in parsed):
        raise ValueError(f"{name} values must be positive")
    return parsed
