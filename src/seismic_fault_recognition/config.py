"""Project configuration helpers for package, CLI and notebook workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import os


@dataclass
class RuntimeConfig:
    """Runtime defaults for local, DataSphere, or CI execution."""

    name: str = "datasphere"
    device: str = "cuda"
    num_workers: int = 2
    seed: int = 42


@dataclass
class PathsConfig:
    """Default project, data, checkpoint, and output paths."""

    project_root: str = "/home/jupyter/project"
    thebe_dir: str = "/home/jupyter/project/Thebe"
    faultseg3d_dir: str = "/home/jupyter/project/FaultSeg3D"
    sr_dir: str = "/home/jupyter/project/SeisGAN_Results"
    checkpoint_dir: str = "/home/jupyter/project/checkpoints"
    output_dir: str = "/home/jupyter/project/outputs"
    test_seis: str = "/home/jupyter/project/Thebe/thebe_test_seis.npz"
    test_fault: str = "/home/jupyter/project/Thebe/thebe_test_fault.npz"


@dataclass
class ClearMLConfig:
    """ClearML project names and optional remote artifact destination."""

    enabled: bool = True
    web_host: str = "https://app.clear.ml"
    project_name: str = "Seismic_Fault_Segmentation"
    pretraining_project_name: str = "Seismic_Pretraining"
    segmentation_project_name: str = "Seismic_Fault_Segmentation"
    sr_project_name: str = "Seismic_Super_Resolution"
    output_uri: str = ""


@dataclass
class TrainingConfig:
    """Common training hyperparameters shared by notebooks and CLI validation."""

    roi_size: tuple[int, int, int] = (128, 128, 128)
    patch_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: int = 1
    learning_rate: float = 1e-4
    max_epochs: int = 100
    threshold: float = 0.5
    overlap: float = 0.5


@dataclass
class ValidationConfig:
    """Validation defaults for threshold sweeps and checkpoint loading policy."""

    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    primary_metric: str = "val_dice_best_threshold"
    checkpoint_strict: bool = True
    sr_metric_normalization: str = "dataset_stats"
    sr_data_range: tuple[float, float] = (-3.0, 3.0)


@dataclass
class ReproducibilityConfig:
    """Seed and deterministic backend settings used by data loaders."""

    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    num_workers: int = 2


@dataclass
class ExperimentConfig:
    """Top-level base configuration assembled from dataclass sections."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    clearml: ClearMLConfig = field(default_factory=ClearMLConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)

    def as_dict(self) -> dict[str, Any]:
        """Return the configuration as a nested dictionary."""

        return asdict(self)

    def ensure_output_dirs(self) -> None:
        """Create configured checkpoint and output directories."""

        for value in (self.paths.checkpoint_dir, self.paths.output_dir):
            Path(value).mkdir(parents=True, exist_ok=True)


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Load project config from YAML and optional nested overrides.

    If PyYAML is not installed, a small parser handles the simple YAML shape
    used by ``configs/datasphere.yaml``.
    """

    config = ExperimentConfig()
    path = _resolve_config_path(config_path)
    if path is not None and path.exists():
        payload = _read_mapping(path)
        assert_valid_config_payload(payload, experiment=False, source=str(path))
        _apply_mapping(config, payload)
    if overrides:
        _apply_mapping(config, overrides)
    return config


def validate_config_file(path: str | os.PathLike[str], *, experiment: bool = False) -> list[str]:
    """Validate one YAML config file and return schema issues."""

    payload = _read_mapping(Path(path))
    return validate_config_payload(payload, experiment=experiment, source=str(path))


def assert_valid_config_file(path: str | os.PathLike[str], *, experiment: bool = False) -> None:
    """Validate one config file and raise ``ValueError`` on schema issues."""

    issues = validate_config_file(path, experiment=experiment)
    if issues:
        raise ValueError("Invalid config:\n" + "\n".join(f"- {issue}" for issue in issues))


def assert_valid_config_payload(
    payload: Mapping[str, Any],
    *,
    experiment: bool = False,
    source: str = "config",
) -> None:
    """Validate an in-memory config payload and raise on schema issues."""

    issues = validate_config_payload(payload, experiment=experiment, source=source)
    if issues:
        raise ValueError("Invalid config:\n" + "\n".join(f"- {issue}" for issue in issues))


def validate_config_payload(
    payload: Mapping[str, Any],
    *,
    experiment: bool = False,
    source: str = "config",
) -> list[str]:
    """Return schema issues for a base or per-experiment config mapping."""

    issues: list[str] = []
    if not isinstance(payload, Mapping):
        return [f"{source}: expected a mapping"]

    if experiment:
        required = (
            "recipe",
            "notebook",
            "stage",
            "dataset",
            "model_variant",
            "loss_profile",
            "augmentation_profile",
            "trainer_profile",
            "data",
            "checkpoints",
            "outputs",
            "training",
        )
        _require_keys(payload, required, source, issues)
    else:
        _require_keys(payload, ("runtime", "paths", "clearml", "training"), source, issues)

    for section in ("runtime", "paths", "clearml", "training", "validation", "reproducibility"):
        if section in payload and not isinstance(payload[section], Mapping):
            issues.append(f"{source}.{section}: expected a mapping")

    training = payload.get("training", {})
    if isinstance(training, Mapping):
        for key in ("roi_size", "patch_size"):
            if key in training:
                _validate_len3(training[key], f"{source}.training.{key}", issues)
        if "threshold" in training:
            _validate_range(training["threshold"], f"{source}.training.threshold", issues, min_value=0.0, max_value=1.0)
        if "overlap" in training:
            _validate_range(training["overlap"], f"{source}.training.overlap", issues, min_value=0.0, max_value=0.999999)
        for key in ("batch_size", "max_epochs", "accumulation_steps"):
            if key in training:
                _validate_positive_int(training[key], f"{source}.training.{key}", issues)
        if "num_workers" in training:
            _validate_positive_int(training["num_workers"], f"{source}.training.num_workers", issues, allow_zero=True)

    validation = payload.get("validation", {})
    if isinstance(validation, Mapping):
        if "thresholds" in validation:
            values = _as_sequence(validation["thresholds"])
            if not values:
                issues.append(f"{source}.validation.thresholds: expected at least one threshold")
            for index, value in enumerate(values):
                _validate_range(value, f"{source}.validation.thresholds[{index}]", issues, min_value=0.0, max_value=1.0)
        if "primary_metric" in validation and not str(validation["primary_metric"]).strip():
            issues.append(f"{source}.validation.primary_metric: expected a non-empty string")
        if "checkpoint_strict" in validation and not isinstance(validation["checkpoint_strict"], bool):
            issues.append(f"{source}.validation.checkpoint_strict: expected a boolean")
        if "sr_metric_normalization" in validation:
            allowed = {"dataset_stats", "fixed_range", "per_volume_minmax", "none"}
            if str(validation["sr_metric_normalization"]) not in allowed:
                issues.append(f"{source}.validation.sr_metric_normalization: expected one of {sorted(allowed)}")
        if "sr_data_range" in validation:
            values = _as_sequence(validation["sr_data_range"])
            if len(values) != 2:
                issues.append(f"{source}.validation.sr_data_range: expected exactly two values")
            else:
                try:
                    if float(values[1]) <= float(values[0]):
                        issues.append(f"{source}.validation.sr_data_range: expected high value greater than low value")
                except (TypeError, ValueError):
                    issues.append(f"{source}.validation.sr_data_range: expected numeric values")

    reproducibility = payload.get("reproducibility", {})
    if isinstance(reproducibility, Mapping):
        if "seed" in reproducibility:
            _validate_int(reproducibility["seed"], f"{source}.reproducibility.seed", issues)
        if "num_workers" in reproducibility:
            _validate_positive_int(reproducibility["num_workers"], f"{source}.reproducibility.num_workers", issues, allow_zero=True)
        for key in ("deterministic", "benchmark"):
            if key in reproducibility and not isinstance(reproducibility[key], bool):
                issues.append(f"{source}.reproducibility.{key}: expected a boolean")

    return issues


def _resolve_config_path(config_path: str | os.PathLike[str] | None) -> Path | None:
    if config_path:
        return Path(config_path).expanduser()
    env_path = os.environ.get("SEISMIC_CONFIG")
    if env_path:
        return Path(env_path).expanduser()

    candidates = [
        Path.cwd() / "configs" / "datasphere.yaml",
        Path.cwd().parent / "configs" / "datasphere.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Expected mapping in {path}")
        return data
    except ImportError:
        return _read_basic_yaml(path)


def _read_basic_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
        return tuple(_parse_scalar(item) for item in items)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _apply_mapping(target: Any, payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _apply_mapping(current, value)
        else:
            setattr(target, key, _coerce_value(current, value))


def _coerce_value(current: Any, value: Any) -> Any:
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _require_keys(payload: Mapping[str, Any], keys: Iterable[str], source: str, issues: list[str]) -> None:
    for key in keys:
        if key not in payload:
            issues.append(f"{source}: missing required key {key!r}")


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _validate_len3(value: Any, path: str, issues: list[str]) -> None:
    values = _as_sequence(value)
    if len(values) != 3:
        issues.append(f"{path}: expected exactly three values")
        return
    for index, item in enumerate(values):
        _validate_positive_int(item, f"{path}[{index}]", issues)


def _validate_int(value: Any, path: str, issues: list[str]) -> None:
    if isinstance(value, bool):
        issues.append(f"{path}: expected an integer")
        return
    try:
        int(value)
    except (TypeError, ValueError):
        issues.append(f"{path}: expected an integer")


def _validate_positive_int(value: Any, path: str, issues: list[str], allow_zero: bool = False) -> None:
    _validate_int(value, path, issues)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return
    if parsed < 0 or (parsed == 0 and not allow_zero):
        floor = "non-negative" if allow_zero else "positive"
        issues.append(f"{path}: expected a {floor} integer")


def _validate_range(value: Any, path: str, issues: list[str], *, min_value: float, max_value: float) -> None:
    if isinstance(value, bool):
        issues.append(f"{path}: expected a number")
        return
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        issues.append(f"{path}: expected a number")
        return
    if not (min_value <= parsed <= max_value):
        issues.append(f"{path}: expected {min_value} <= value <= {max_value}")
