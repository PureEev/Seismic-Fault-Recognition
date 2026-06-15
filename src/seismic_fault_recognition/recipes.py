"""Experiment recipe registry.

Recipes make every notebook explicit: each stage selects its own dataset,
augmentations, loss profile, trainer profile and Swin/Omni/SR architecture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentRecipe:
    """Registered package-first experiment recipe."""

    name: str
    notebook: str
    stage: str
    dataset: str
    model_variant: str
    loss_profile: str
    augmentation_profile: str
    trainer_profile: str
    config_name: str
    clearml_task_name: str
    role: str = "primary"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


@dataclass(frozen=True)
class PathCheck:
    """Existence check result for one configured path."""

    key: str
    path: str
    exists: bool


def list_recipes() -> tuple[str, ...]:
    """Return registered recipe names."""

    return tuple(RECIPES)


def get_recipe(name: str) -> ExperimentRecipe:
    """Return a registered recipe by name.

    Raises:
        KeyError: If the recipe is not registered.
    """

    try:
        return RECIPES[name]
    except KeyError as exc:
        known = ", ".join(sorted(RECIPES))
        raise KeyError(f"Unknown recipe {name!r}. Known recipes: {known}") from exc


def recipes_for_stage(stage: str) -> tuple[ExperimentRecipe, ...]:
    """Return recipes whose ``stage`` matches the requested value."""

    return tuple(recipe for recipe in RECIPES.values() if recipe.stage == stage)


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one experiment YAML config.

    Args:
        path: YAML file path.

    Returns:
        Experiment config mapping, or an empty dict when the file is absent.
    """

    path = Path(path)
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return {}
        from .config import assert_valid_config_payload

        assert_valid_config_payload(data, experiment=True, source=str(path))
        return data
    except ImportError:
        data = _read_basic_yaml(path)
        from .config import assert_valid_config_payload

        assert_valid_config_payload(data, experiment=True, source=str(path))
        return data


def validate_recipe_paths(base_config: Any, experiment_config: Mapping[str, Any]) -> list[PathCheck]:
    """Check whether data, output and checkpoint paths referenced by a recipe exist."""

    paths: dict[str, Any] = {}
    cfg_paths = getattr(base_config, "paths", None)
    if cfg_paths is not None:
        for key in (
            "thebe_dir",
            "faultseg3d_dir",
            "sr_dir",
            "checkpoint_dir",
            "output_dir",
            "test_seis",
            "test_fault",
        ):
            if hasattr(cfg_paths, key):
                paths[key] = getattr(cfg_paths, key)
    for section_name in ("data", "checkpoints", "outputs"):
        section = experiment_config.get(section_name, {})
        if isinstance(section, Mapping):
            paths.update(
                {str(key): value for key, value in section.items() if isinstance(value, str) and value.strip()}
            )
    return [PathCheck(key, str(value), Path(str(value)).expanduser().exists()) for key, value in sorted(paths.items())]


def describe_recipe_components(recipe_name: str) -> dict[str, Any]:
    """Return recipe metadata plus selected loss, augmentation, trainer and model profiles."""

    recipe = get_recipe(recipe_name)
    return {
        "recipe": recipe.as_dict(),
        "loss_profile": _safe_profile("loss", recipe.loss_profile),
        "augmentation_profile": _safe_profile("augmentation", recipe.augmentation_profile),
        "trainer_profile": _safe_profile("trainer", recipe.trainer_profile),
        "model_variant": _safe_profile("model", recipe.model_variant),
    }


def _safe_profile(kind: str, name: str) -> dict[str, Any]:
    try:
        if kind == "loss":
            from .registry import LOSS_REGISTRY
            if name not in LOSS_REGISTRY:
                return {"name": name, "error": "Not registered"}
            return {"name": name}
        if kind == "augmentation":
            from .registry import AUGMENTATION_REGISTRY
            if name not in AUGMENTATION_REGISTRY:
                return {"name": name, "error": "Not registered"}
            return {"name": name}
        if kind == "trainer":
            from .registry import TRAINER_REGISTRY
            if name not in TRAINER_REGISTRY:
                return {"name": name, "error": "Not registered"}
            profile = TRAINER_REGISTRY.get(name)
            return {
                "name": profile.name,
                "stage": profile.stage,
            }
        if kind == "model":
            from .registry import MODEL_REGISTRY
            if name not in MODEL_REGISTRY:
                return {"name": name, "error": "Not registered"}
            return {"name": name}
    except Exception as exc:
        return {"name": name, "error": str(exc)}
    return {"name": name}


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
        return [_parse_scalar(item.strip()) for item in value[1:-1].split(",") if item.strip()]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


RECIPES: dict[str, ExperimentRecipe] = {
    "environment_and_data_check": ExperimentRecipe(
        name="environment_and_data_check",
        notebook="00_environment_and_data_check.ipynb",
        stage="setup",
        dataset="all",
        model_variant="none",
        loss_profile="segmentation_bce",
        augmentation_profile="thebe_random_crop",
        trainer_profile="segmentation_validation",
        config_name="00_environment_and_data_check.yaml",
        clearml_task_name="Environment_Data_Check",
    ),
    "data_cleaning_and_audit": ExperimentRecipe(
        name="data_cleaning_and_audit",
        notebook="01_data_cleaning_and_audit.ipynb",
        stage="data_audit",
        dataset="thebe",
        model_variant="none",
        loss_profile="segmentation_bce",
        augmentation_profile="thebe_random_crop",
        trainer_profile="segmentation_validation",
        config_name="01_data_cleaning_and_audit.yaml",
        clearml_task_name="Thebe_Data_Audit",
    ),
    "simmim_swinunetr_thebe_pretrain": ExperimentRecipe(
        name="simmim_swinunetr_thebe_pretrain",
        notebook="02_simmim_swinunetr_thebe_pretrain.ipynb",
        stage="simmim_pretrain",
        dataset="thebe",
        model_variant="swin_tiny",
        loss_profile="simmim_l1",
        augmentation_profile="simmim_masking",
        trainer_profile="simmim_pretrain",
        config_name="02_simmim_swinunetr_thebe_pretrain.yaml",
        clearml_task_name="SimMIM_SwinUNETR_Tiny",
    ),
    "simmim_omniseis_thebe_pretrain": ExperimentRecipe(
        name="simmim_omniseis_thebe_pretrain",
        notebook="03_simmim_omniseis_thebe_pretrain.ipynb",
        stage="simmim_pretrain",
        dataset="thebe",
        model_variant="omniseis_encoder",
        loss_profile="simmim_l1",
        augmentation_profile="simmim_masking",
        trainer_profile="simmim_pretrain",
        config_name="03_simmim_omniseis_thebe_pretrain.yaml",
        clearml_task_name="SimMIM_SeisOmni",
    ),
    "faultseg3d_swin_tiny_pretrain": ExperimentRecipe(
        name="faultseg3d_swin_tiny_pretrain",
        notebook="04_faultseg3d_swin_tiny_pretrain.ipynb",
        stage="faultseg3d_pretrain",
        dataset="faultseg3d",
        model_variant="swin_tiny",
        loss_profile="faultseg3d_combined_sym",
        augmentation_profile="faultseg3d_train",
        trainer_profile="faultseg3d_pretrain",
        config_name="04_faultseg3d_swin_tiny_pretrain.yaml",
        clearml_task_name="Swin_tiny_Modified_Loaded",
    ),
    "faultseg3d_omniseis_pretrain": ExperimentRecipe(
        name="faultseg3d_omniseis_pretrain",
        notebook="05_faultseg3d_omniseis_pretrain.ipynb",
        stage="faultseg3d_pretrain",
        dataset="faultseg3d",
        model_variant="omniseis_segmentation",
        loss_profile="faultseg3d_combined_sym",
        augmentation_profile="faultseg3d_train",
        trainer_profile="faultseg3d_pretrain",
        config_name="05_faultseg3d_omniseis_pretrain.yaml",
        clearml_task_name="OmniSeis_Pretrain_Original_Decoder",
    ),
    "swinunetr_thebe_finetune_raw": ExperimentRecipe(
        name="swinunetr_thebe_finetune_raw",
        notebook="06_swinunetr_thebe_finetune_raw.ipynb",
        stage="thebe_finetune",
        dataset="thebe_raw",
        model_variant="swin_tiny",
        loss_profile="swinunetr_monai_dice",
        augmentation_profile="thebe_random_crop",
        trainer_profile="thebe_finetune",
        config_name="06_swinunetr_thebe_finetune_raw.yaml",
        clearml_task_name="SwinUNETR_Thebe_Raw",
    ),
    "swinunetr_thebe_finetune_clean_sr_cubes": ExperimentRecipe(
        name="swinunetr_thebe_finetune_clean_sr_cubes",
        notebook="07_swinunetr_thebe_finetune_clean_sr_cubes.ipynb",
        stage="thebe_finetune",
        dataset="thebe_clean_sr",
        model_variant="swin_tiny",
        loss_profile="thebe_combined_sym",
        augmentation_profile="thebe_random_crop",
        trainer_profile="thebe_finetune",
        config_name="07_swinunetr_thebe_finetune_clean_sr_cubes.yaml",
        clearml_task_name="SwinTiny_FineTune_Thebe",
    ),
    "omniseis_thebe_finetune_raw": ExperimentRecipe(
        name="omniseis_thebe_finetune_raw",
        notebook="08_omniseis_thebe_finetune_raw.ipynb",
        stage="thebe_finetune",
        dataset="thebe_raw",
        model_variant="omniseis_segmentation",
        loss_profile="thebe_combined_sym",
        augmentation_profile="thebe_random_crop",
        trainer_profile="thebe_finetune",
        config_name="08_omniseis_thebe_finetune_raw.yaml",
        clearml_task_name="OmniSeis_Pretrained_Seg_Raw_cubes",
    ),
    "omniseis_thebe_finetune_clean_cubes": ExperimentRecipe(
        name="omniseis_thebe_finetune_clean_cubes",
        notebook="09_omniseis_thebe_finetune_clean_cubes.ipynb",
        stage="thebe_finetune",
        dataset="thebe_clean",
        model_variant="omniseis_segmentation",
        loss_profile="thebe_combined_sym",
        augmentation_profile="thebe_random_crop",
        trainer_profile="thebe_finetune",
        config_name="09_omniseis_thebe_finetune_clean_cubes.yaml",
        clearml_task_name="OmniSeis_Pretrained_Seg_Clean_cubes",
    ),
    "omniseis_thebe_finetune_clean_sr_aug_reg": ExperimentRecipe(
        name="omniseis_thebe_finetune_clean_sr_aug_reg",
        notebook="10_omniseis_thebe_finetune_clean_sr_aug_reg.ipynb",
        stage="thebe_finetune",
        dataset="thebe_clean_sr",
        model_variant="omniseis_segmentation",
        loss_profile="thebe_combined_sym",
        augmentation_profile="thebe_random_crop",
        trainer_profile="thebe_finetune",
        config_name="10_omniseis_thebe_finetune_clean_sr_aug_reg.yaml",
        clearml_task_name="OmniSeis_Pretrained_Seg_Raw_cubes_aug_reg_sr",
    ),
    "sr_training_seisgan": ExperimentRecipe(
        name="sr_training_seisgan",
        notebook="11_sr_training_seisgan.ipynb",
        stage="sr_training",
        dataset="thebe_sr",
        model_variant="omniseis_sr_generator",
        loss_profile="sr_l1_vgg_gan",
        augmentation_profile="sr_degradation",
        trainer_profile="sr_training",
        config_name="11_sr_training_seisgan.yaml",
        clearml_task_name="3D_SeisGAN_Training",
    ),
    "segmentation_validation": ExperimentRecipe(
        name="segmentation_validation",
        notebook="12_segmentation_validation.ipynb",
        stage="validation",
        dataset="thebe_test",
        model_variant="swin_tiny",
        loss_profile="segmentation_bce",
        augmentation_profile="thebe_random_crop",
        trainer_profile="segmentation_validation",
        config_name="12_segmentation_validation.yaml",
        clearml_task_name="Segmentation_Validation",
    ),
    "end_to_end_sr_segmentation_validation": ExperimentRecipe(
        name="end_to_end_sr_segmentation_validation",
        notebook="13_end_to_end_sr_segmentation_validation.ipynb",
        stage="validation",
        dataset="thebe_test_sr",
        model_variant="sr_plus_segmentation",
        loss_profile="segmentation_bce",
        augmentation_profile="sr_degradation",
        trainer_profile="segmentation_validation",
        config_name="13_end_to_end_sr_segmentation_validation.yaml",
        clearml_task_name="SR_Segmentation_Validation",
    ),
    "swinunetr_tiny": ExperimentRecipe(
        name="swinunetr_tiny",
        notebook="14_swinunetr_tiny.ipynb",
        stage="architecture_check",
        dataset="none",
        model_variant="swin_tiny",
        loss_profile="faultseg3d_combined_sym",
        augmentation_profile="faultseg3d_val",
        trainer_profile="faultseg3d_pretrain",
        config_name="14_swinunetr_tiny.yaml",
        clearml_task_name="SwinUNETR_Tiny_Final",
        role="reference",
    ),
    "faultformer_variants": ExperimentRecipe(
        name="faultformer_variants",
        notebook="15_faultformer_variants.ipynb",
        stage="architecture_variants",
        dataset="none",
        model_variant="faultformer_attention",
        loss_profile="faultseg3d_combined_sym",
        augmentation_profile="faultseg3d_val",
        trainer_profile="faultseg3d_pretrain",
        config_name="15_faultformer_variants.yaml",
        clearml_task_name="FaultFormer_Variants",
        role="experimental",
    ),
    "visualization_3d": ExperimentRecipe(
        name="visualization_3d",
        notebook="16_3d_visualization.ipynb",
        stage="visualization",
        dataset="predictions",
        model_variant="none",
        loss_profile="segmentation_bce",
        augmentation_profile="thebe_random_crop",
        trainer_profile="segmentation_validation",
        config_name="16_3d_visualization.yaml",
        clearml_task_name="3D_Visualization",
    ),
}
