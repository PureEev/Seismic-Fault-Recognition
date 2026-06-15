"""Dataset and array utilities used by the training notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import itertools
import json
import random

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - DataSphere normally has torch.
    torch = None  # type: ignore

    class Dataset:  # type: ignore
        """Fallback dataset base class used when PyTorch is unavailable."""

        pass


ArraySource = str | Path | Mapping[str, np.ndarray]


def normalize_zscore(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a float32 z-score normalized copy of a 3D volume.

    Args:
        volume: Input array. NaNs are replaced by finite values.
        eps: Small denominator guard for nearly constant volumes.

    Returns:
        Array with approximately zero mean and unit standard deviation.
    """

    arr = np.array(volume, dtype=np.float32, copy=True)
    np.nan_to_num(arr, copy=False)
    return (arr - float(arr.mean())) / (float(arr.std()) + eps)


def pad_to_min_shape(
    array: np.ndarray,
    min_shape: Sequence[int],
    mode: str = "constant",
) -> np.ndarray:
    """Pad the last three dimensions up to ``min_shape``.

    Args:
        array: Input array with at least three dimensions.
        min_shape: Minimum ``(D, H, W)`` shape for the trailing axes.
        mode: NumPy padding mode.

    Returns:
        Original array if no padding is needed, otherwise a padded array.
    """

    pads = []
    for current, target in zip(array.shape[-3:], min_shape):
        pads.append((0, max(0, int(target) - int(current))))
    if not any(right for _, right in pads):
        return array
    prefix = [(0, 0)] * (array.ndim - 3)
    return np.pad(array, prefix + pads, mode=mode)


def crop_3d(
    array: np.ndarray,
    target_shape: Sequence[int],
    random_crop: bool = False,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Pad if needed and crop one 3D array to ``target_shape``.

    Args:
        array: Input array, using the last three axes as spatial axes.
        target_shape: Desired spatial shape.
        random_crop: Whether to sample a random crop instead of center crop.
        rng: Optional Python RNG for deterministic random crops.

    Returns:
        Cropped array with trailing shape equal to ``target_shape``.
    """

    slices = crop_slices(array.shape[-3:], target_shape, random_crop=random_crop, rng=rng)
    padded = pad_to_min_shape(array, target_shape)
    return padded[(...,) + slices]


def crop_slices(
    shape: Sequence[int],
    target_shape: Sequence[int],
    random_crop: bool = False,
    rng: random.Random | None = None,
) -> tuple[slice, slice, slice]:
    """Compute center or random crop slices for a 3D shape.

    Args:
        shape: Source spatial shape.
        target_shape: Requested crop shape.
        random_crop: Whether to sample valid random origins.
        rng: Optional Python RNG.

    Returns:
        Three slices for the spatial axes.
    """

    target = tuple(int(x) for x in target_shape)
    rng = rng or random
    starts: list[int] = []
    padded_shape = tuple(max(int(dim), int(size)) for dim, size in zip(shape, target))
    for dim, size in zip(padded_shape, target):
        max_start = int(dim) - int(size)
        if max_start <= 0:
            starts.append(0)
        elif random_crop:
            starts.append(rng.randint(0, max_start))
        else:
            starts.append(max_start // 2)
    z, y, x = starts
    dz, dy, dx = target
    return slice(z, z + dz), slice(y, y + dy), slice(x, x + dx)


def apply_crop_slices(array: np.ndarray, target_shape: Sequence[int], slices: tuple[slice, slice, slice]) -> np.ndarray:
    """Apply previously computed crop slices after padding to ``target_shape``."""

    padded = pad_to_min_shape(array, target_shape)
    return padded[(...,) + slices]


def load_npz_mapping(source: ArraySource, mmap_mode: str | None = "r") -> Mapping[str, np.ndarray]:
    """Return a mapping-like NPZ object or pass through an in-memory mapping.

    Args:
        source: NPZ file path or already loaded mapping.
        mmap_mode: NumPy mmap mode for file-backed NPZ loading.

    Returns:
        Mapping from array keys to NumPy arrays.
    """

    if isinstance(source, Mapping):
        return source
    return np.load(str(source), mmap_mode=mmap_mode)


def analyze_npz(source: str | Path) -> list[dict[str, Any]]:
    """Return key, shape, dtype and dimensionality metadata for one NPZ file."""

    with np.load(source, mmap_mode="r") as data:
        rows = []
        for key in data.files:
            arr = data[key]
            rows.append(
                {
                    "key": key,
                    "shape": tuple(int(x) for x in arr.shape),
                    "dtype": str(arr.dtype),
                    "ndim": int(arr.ndim),
                    "is_3d": bool(arr.ndim == 3),
                }
            )
    return rows


def audit_npz_pair(
    seis_source: str | Path,
    fault_source: str | Path,
    split_name: str,
    max_arrays: int | None = None,
) -> dict[str, Any]:
    """Audit paired seismic/fault NPZ files for shape, sparsity and empty masks."""

    with np.load(seis_source, mmap_mode="r") as seis, np.load(fault_source, mmap_mode="r") as fault:
        seis_keys = set(seis.files)
        fault_keys = set(fault.files)
        common = sorted(seis_keys.intersection(fault_keys))
        selected = common[: int(max_arrays)] if max_arrays else common
        rows = []
        for key in selected:
            seis_arr = np.asarray(seis[key])
            fault_arr = np.asarray(fault[key])
            fault_bool = fault_arr > 0
            rows.append(
                {
                    "key": key,
                    "seis_shape": [int(item) for item in seis_arr.shape],
                    "fault_shape": [int(item) for item in fault_arr.shape],
                    "seis_dtype": str(seis_arr.dtype),
                    "fault_dtype": str(fault_arr.dtype),
                    "shape_match": bool(seis_arr.shape == fault_arr.shape),
                    "seis_nan_fraction": float(np.isnan(seis_arr).mean()) if np.issubdtype(seis_arr.dtype, np.floating) else 0.0,
                    "seis_zero_fraction": float((seis_arr == 0).mean()),
                    "seis_mean": float(np.nanmean(seis_arr)),
                    "seis_std": float(np.nanstd(seis_arr)),
                    "fault_positive_fraction": float(fault_bool.mean()),
                    "fault_empty": bool(not fault_bool.any()),
                }
            )
    positive_fractions = [row["fault_positive_fraction"] for row in rows]
    return {
        "split_name": split_name,
        "seis_path": str(seis_source),
        "fault_path": str(fault_source),
        "seis_key_count": len(seis_keys),
        "fault_key_count": len(fault_keys),
        "paired_key_count": len(common),
        "paired_keys": common,
        "audited_key_count": len(rows),
        "missing_fault_keys": sorted(seis_keys - fault_keys),
        "missing_seis_keys": sorted(fault_keys - seis_keys),
        "empty_fault_count": int(sum(1 for row in rows if row["fault_empty"])),
        "mean_fault_positive_fraction": float(np.mean(positive_fractions)) if positive_fractions else 0.0,
        "rows": rows,
    }


def build_npz_split_audit(
    splits: Mapping[str, tuple[str | Path, str | Path]],
    max_arrays: int | None = None,
) -> dict[str, Any]:
    """Build a multi-split audit report and detect key leakage between splits."""

    reports = {
        split_name: audit_npz_pair(seis_path, fault_path, split_name, max_arrays=max_arrays)
        for split_name, (seis_path, fault_path) in splits.items()
    }
    key_sets = {name: set(report["paired_keys"]) for name, report in reports.items()}
    intersections = []
    for left, right in itertools.combinations(sorted(key_sets), 2):
        common = sorted(key_sets[left].intersection(key_sets[right]))
        intersections.append({"left": left, "right": right, "count": len(common), "keys": common[:100]})
    failed = any(item["count"] > 0 for item in intersections)
    failed = failed or any(report["missing_fault_keys"] or report["missing_seis_keys"] for report in reports.values())
    failed = failed or any(
        not row["shape_match"] for report in reports.values() for row in report["rows"]
    )
    return {"status": "failed" if failed else "ok", "splits": reports, "split_key_intersections": intersections}


def save_json_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Save an audit/report mapping as formatted JSON.

    Args:
        path: Output JSON path.
        report: JSON-serializable report mapping.

    Returns:
        Resolved output path object.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def paired_keys(seis_source: ArraySource, fault_source: ArraySource) -> list[str]:
    """Return sorted keys present in both seismic and fault sources."""

    seis = load_npz_mapping(seis_source)
    fault = load_npz_mapping(fault_source)
    try:
        return sorted(set(seis.keys()).intersection(fault.keys()))
    finally:
        close_array_source(seis)
        close_array_source(fault)


def close_array_source(source: Any) -> None:
    """Close a mapping-like source if it exposes a ``close`` method."""

    close = getattr(source, "close", None)
    if callable(close):
        close()


def resolve_local_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a local filesystem path without downloading anything."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute() or base_dir is None:
        return candidate
    return Path(base_dir).expanduser() / candidate


def require_paths(paths: Mapping[str, str | Path], base_dir: str | Path | None = None) -> dict[str, Path]:
    """Resolve paths and raise a clear error if any required local file is missing."""

    resolved = {name: resolve_local_path(path, base_dir) for name, path in paths.items() if str(path)}
    missing = {name: path for name, path in resolved.items() if not path.exists()}
    if missing:
        details = "\n".join(f"- {name}: {path}" for name, path in missing.items())
        raise FileNotFoundError(f"Required local data/checkpoint paths are missing:\n{details}")
    return resolved


def discover_faultseg3d_memmaps(
    seis_dir: str | Path,
    fault_dir: str | Path,
    seis_pattern: str = "*.dat",
    fault_pattern: str = "*.dat",
) -> tuple[dict[int, Path], dict[int, Path]]:
    """Discover FaultSeg3D memmap pairs by numeric id in filenames."""

    import re

    def collect(directory: str | Path, pattern: str) -> dict[int, Path]:
        """Collect memmap files by numeric id parsed from filenames."""

        result: dict[int, Path] = {}
        for path in sorted(Path(directory).expanduser().glob(pattern)):
            match = re.search(r"(\d+)", path.stem)
            if match:
                result[int(match.group(1))] = path
        return result

    seis_map = collect(seis_dir, seis_pattern)
    fault_map = collect(fault_dir, fault_pattern)
    common = sorted(set(seis_map).intersection(fault_map))
    if not common:
        raise ValueError(f"No matching FaultSeg3D memmap ids found in {seis_dir} and {fault_dir}")
    return ({idx: seis_map[idx] for idx in common}, {idx: fault_map[idx] for idx in common})


@dataclass
class DatasetSample:
    """Container for one seismic sample and optional target metadata."""

    seismic: np.ndarray
    fault: np.ndarray | None = None
    key: str | None = None


class SeisFaultDataset(Dataset):
    """Pairs seismic and fault 3D arrays from ``.npz`` files or mappings."""

    def __init__(
        self,
        seis_source: ArraySource,
        fault_source: ArraySource | None = None,
        target_shape: Sequence[int] = (128, 128, 128),
        random_crop: bool = True,
        normalize_seis: bool = True,
        seed: int = 42,
        pairs: Sequence[str] | None = None,
        return_meta: bool = False,
    ) -> None:
        self.seis_source = seis_source
        self.fault_source = fault_source
        self.target_shape = tuple(int(x) for x in target_shape)
        self.random_crop = random_crop
        self.normalize_seis = normalize_seis
        self.return_meta = return_meta
        self._rng = random.Random(seed)
        self._seis_data = load_npz_mapping(seis_source)
        self._fault_data = load_npz_mapping(fault_source) if fault_source is not None else None

        if pairs is None:
            if self._fault_data is None:
                pairs = sorted(self._seis_data.keys())
            else:
                pairs = sorted(set(self._seis_data.keys()).intersection(self._fault_data.keys()))
        if not pairs:
            raise ValueError("No paired keys found for seismic/fault sources")
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Any:
        key = self.pairs[index]
        raw_seis = np.asarray(self._seis_data[key])
        slices = crop_slices(raw_seis.shape[-3:], self.target_shape, self.random_crop, self._rng)
        seis = apply_crop_slices(raw_seis, self.target_shape, slices)
        if self.normalize_seis:
            seis = normalize_zscore(seis)
        fault = None
        if self._fault_data is not None:
            fault = apply_crop_slices(np.asarray(self._fault_data[key]), self.target_shape, slices)
            fault = (fault > 0).astype(np.float32, copy=False)

        if torch is not None:
            seis_out: Any = torch.from_numpy(seis.astype(np.float32, copy=False)).unsqueeze(0)
            fault_out: Any = (
                torch.from_numpy(fault.astype(np.float32, copy=False)).unsqueeze(0)
                if fault is not None
                else None
            )
        else:
            seis_out = seis.astype(np.float32, copy=False)[None]
            fault_out = fault.astype(np.float32, copy=False)[None] if fault is not None else None

        if self.return_meta:
            return {"seismic": seis_out, "fault": fault_out, "key": key}
        if fault_out is None:
            return seis_out
        return seis_out, fault_out

    def close(self) -> None:
        """Close file-backed NPZ sources held by the dataset."""

        close_array_source(self._seis_data)
        close_array_source(self._fault_data)

    def __del__(self) -> None:
        self.close()


class FaultSeg3DMemmapDataset(Dataset):
    """Reads indexed FaultSeg3D memmap pairs from filename maps."""

    def __init__(
        self,
        seis_map: Mapping[int, str | Path],
        fault_map: Mapping[int, str | Path],
        shape: Sequence[int] = (128, 128, 128),
        dtype: str | np.dtype = "float32",
        normalize: bool = True,
    ) -> None:
        self.seis_map = dict(seis_map)
        self.fault_map = dict(fault_map)
        self.indices = sorted(set(self.seis_map).intersection(self.fault_map))
        if not self.indices:
            raise ValueError("No matching memmap indices found")
        self.shape = tuple(int(x) for x in shape)
        self.dtype = np.dtype(dtype)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        file_index = self.indices[index]
        seis = np.memmap(self.seis_map[file_index], dtype=self.dtype, mode="r").reshape(self.shape)
        fault = np.memmap(self.fault_map[file_index], dtype=self.dtype, mode="r").reshape(self.shape)
        seis_arr = normalize_zscore(np.asarray(seis)) if self.normalize else np.asarray(seis, dtype=np.float32)
        fault_arr = (np.asarray(fault) > 0).astype(np.float32)
        if torch is None:
            return seis_arr[None], fault_arr[None]
        return torch.from_numpy(seis_arr.copy()).unsqueeze(0), torch.from_numpy(fault_arr.copy()).unsqueeze(0)


class SeismicPrecroppedDataset(Dataset):
    """Single-source 3D seismic dataset for SimMIM-style reconstruction."""

    def __init__(
        self,
        seis_source: ArraySource,
        target_shape: Sequence[int] = (128, 128, 128),
        random_crop: bool = True,
        normalize: bool = True,
        seed: int = 42,
        keys: Sequence[str] | None = None,
    ) -> None:
        self.source = seis_source
        self.target_shape = tuple(int(x) for x in target_shape)
        self.random_crop = random_crop
        self.normalize = normalize
        self._rng = random.Random(seed)
        self._data = load_npz_mapping(seis_source)
        self.keys = list(keys) if keys is not None else sorted(self._data.keys())
        if not self.keys:
            raise ValueError("No seismic cubes found")

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Any:
        key = self.keys[index]
        cube = crop_3d(np.asarray(self._data[key]), self.target_shape, self.random_crop, self._rng)
        if self.normalize:
            cube = normalize_zscore(cube)
        if torch is None:
            return cube.astype(np.float32, copy=False)[None]
        return torch.from_numpy(cube.astype(np.float32, copy=False)).unsqueeze(0)

    def close(self) -> None:
        """Close the file-backed NPZ source held by the dataset."""

        close_array_source(self._data)

    def __del__(self) -> None:
        self.close()


class MaskGenerator3D:
    """Generate random cubic binary masks for 3D SimMIM pretraining."""

    def __init__(
        self,
        input_size: Sequence[int] = (128, 128, 128),
        mask_patch_size: int = 16,
        mask_ratio: float = 0.6,
        seed: int | None = None,
    ) -> None:
        self.input_size = tuple(int(x) for x in input_size)
        self.mask_patch_size = int(mask_patch_size)
        self.mask_ratio = float(mask_ratio)
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> Any:
        grid = tuple(max(1, size // self.mask_patch_size) for size in self.input_size)
        total = int(np.prod(grid))
        masked = int(round(total * self.mask_ratio))
        values = np.concatenate(
            [np.ones(masked, dtype=np.float32), np.zeros(total - masked, dtype=np.float32)]
        )
        self.rng.shuffle(values)
        mask = values.reshape(grid)
        mask = np.kron(mask, np.ones((self.mask_patch_size,) * 3, dtype=np.float32))
        mask = mask[: self.input_size[0], : self.input_size[1], : self.input_size[2]]
        if torch is None:
            return mask[None]
        return torch.from_numpy(mask).unsqueeze(0)


class SimMIMDatasetWrapper(Dataset):
    """Wrap a seismic dataset and return ``(masked, target, mask)`` samples."""

    def __init__(self, base_dataset: Dataset, mask_generator: MaskGenerator3D, mask_token_value: float = 0.0):
        """Create a masked reconstruction dataset wrapper.

        Args:
            base_dataset: Dataset returning seismic tensors or tuples whose
                first item is a seismic tensor.
            mask_generator: Callable returning a binary mask shaped like one sample.
            mask_token_value: Value written into masked voxels.
        """

        self.base_dataset = base_dataset
        self.mask_generator = mask_generator
        self.mask_token_value = float(mask_token_value)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Any:
        sample = self.base_dataset[index]
        seismic = sample[0] if isinstance(sample, tuple) else sample
        mask = self.mask_generator()
        masked = seismic.clone() if hasattr(seismic, "clone") else seismic.copy()
        masked[mask == 1] = self.mask_token_value
        return masked, seismic, mask


class SRDynamicDataset(Dataset):
    """On-the-fly LR/HR seismic patches for super-resolution training."""

    def __init__(
        self,
        hr_source: ArraySource,
        degrade_fn: Callable[[np.ndarray], np.ndarray],
        patch_size: Sequence[int] = (128, 128, 128),
        random_crop: bool = True,
        normalize: bool = True,
        seed: int = 42,
        keys: Sequence[str] | None = None,
    ) -> None:
        self.hr_source = hr_source
        self.degrade_fn = degrade_fn
        self.patch_size = tuple(int(x) for x in patch_size)
        self.random_crop = random_crop
        self.normalize = normalize
        self._rng = random.Random(seed)
        self._data = load_npz_mapping(hr_source)
        self.keys = list(keys) if keys is not None else sorted(self._data.keys())
        if not self.keys:
            raise ValueError("No HR seismic cubes found")

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Any:
        key = self.keys[index]
        hr = crop_3d(np.asarray(self._data[key]), self.patch_size, self.random_crop, self._rng)
        lr = self.degrade_fn(hr)
        if self.normalize:
            hr = normalize_zscore(hr)
            lr = normalize_zscore(lr)
        hr = hr.astype(np.float32, copy=False)
        lr = lr.astype(np.float32, copy=False)
        if torch is None:
            return lr[None], hr[None]
        return torch.from_numpy(lr).unsqueeze(0), torch.from_numpy(hr).unsqueeze(0)

    def close(self) -> None:
        """Close the file-backed HR NPZ source held by the dataset."""

        close_array_source(self._data)

    def __del__(self) -> None:
        self.close()
