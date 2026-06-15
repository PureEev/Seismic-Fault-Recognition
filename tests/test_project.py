from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import sys
import unittest
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seismic_fault_recognition.config import load_config

try:
    import numpy as np
except ImportError:  # pragma: no cover - local minimal Python image.
    np = None  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover - local minimal Python image.
    torch = None  # type: ignore


class ConfigTest(unittest.TestCase):
    def test_load_default_config(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.runtime.name, "datasphere")
        self.assertEqual(tuple(cfg.training.roi_size), (128, 128, 128))
        self.assertEqual(cfg.validation.primary_metric, "val_dice_best_threshold")

    def test_configs_validate(self) -> None:
        from seismic_fault_recognition.config import validate_config_file, validate_config_payload

        root = Path(__file__).resolve().parents[1]
        self.assertEqual(validate_config_file(root / "configs" / "datasphere.yaml"), [])
        for path in sorted((root / "configs" / "experiments").glob("*.yaml")):
            self.assertEqual(validate_config_file(path, experiment=True), [], path.name)

        issues = validate_config_payload({"runtime": {}, "paths": {}, "clearml": {}, "training": {"threshold": 2.0}})
        self.assertTrue(any("threshold" in issue for issue in issues))


class RegistryTest(unittest.TestCase):
    def test_recipe_registry_has_extended_set(self) -> None:
        from seismic_fault_recognition.recipes import get_recipe, list_recipes

        recipes = list_recipes()
        self.assertEqual(len(recipes), 17)
        self.assertEqual(get_recipe("simmim_swinunetr_thebe_pretrain").trainer_profile, "simmim_pretrain")
        self.assertEqual(get_recipe("sr_training_seisgan").loss_profile, "sr_l1_vgg_gan")

    def test_loss_augmentation_trainer_registries(self) -> None:
        from seismic_fault_recognition.augmentations import get_augmentation_profile, list_augmentation_profiles
        from seismic_fault_recognition.losses import get_loss_profile, list_loss_profiles
        from seismic_fault_recognition.trainers import get_trainer_profile, list_trainer_profiles

        self.assertIn("faultseg3d_train", list_augmentation_profiles())
        self.assertIn("thebe_random_crop", list_augmentation_profiles())
        self.assertEqual(get_augmentation_profile("simmim_masking").stage, "simmim_pretrain")
        self.assertIn("faultseg3d_combined_sym", list_loss_profiles())
        self.assertEqual(get_loss_profile("thebe_stable_combined").weights["focal"], 0.6)
        self.assertIn("sr_training", list_trainer_profiles())
        self.assertEqual(get_trainer_profile("thebe_finetune").stage, "thebe_finetune")
        self.assertFalse(hasattr(get_trainer_profile("thebe_finetune"), "source_" + "notebooks"))

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_single_final_swin_tiny_registry_and_shape_contract(self) -> None:
        try:
            import monai  # noqa: F401
        except ImportError:
            self.skipTest("MONAI is not installed in this environment")

        from seismic_fault_recognition.models.factory import build_model_by_name
        from seismic_fault_recognition.models.swinunetr_variants import (
            SWIN_TINY_FEATURE_SIZE,
            SWIN_TINY_INPUT_SIZE,
        )
        from seismic_fault_recognition.registry import MODEL_REGISTRY

        legacy_names = {
            "monai_baseline",
            "swin_monai_baseline",
            "modified_patch5_nodeep",
            "swin_modified_patch5_nodeep",
            "swin_tiny_wrapper",
            "custom_p5_nodeep",
            "swin_custom_p5_nodeep",
        }
        self.assertIn("swin_tiny", MODEL_REGISTRY)
        self.assertTrue(legacy_names.isdisjoint(MODEL_REGISTRY.list()))
        self.assertEqual(SWIN_TINY_INPUT_SIZE, (128, 128, 128))
        self.assertEqual(SWIN_TINY_FEATURE_SIZE, 48)

        with self.assertRaisesRegex(ValueError, "requires img_size"):
            build_model_by_name("swin_tiny", img_size=(64, 128, 128))

        model = build_model_by_name("swin_tiny", use_checkpoint=False)
        self.assertEqual(tuple(model.swinViT.patch_embed.proj.weight.shape), (48, 1, 5, 5, 5))
        self.assertEqual(tuple(model.enc2to3.weight.shape), (96, 48, 1, 1, 1))
        self.assertEqual(tuple(model.enc3to4.weight.shape), (192, 96, 1, 1, 1))
        self.assertFalse(hasattr(model, "encoder10"))
        self.assertFalse(hasattr(model, "decoder5"))

    def test_final_notebooks_are_clean(self) -> None:
        import json
        import re

        notebooks = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(notebooks), 17)
        training_notebooks = {
            "02_simmim_swinunetr_thebe_pretrain.ipynb",
            "03_simmim_omniseis_thebe_pretrain.ipynb",
            "04_faultseg3d_swin_tiny_pretrain.ipynb",
            "05_faultseg3d_omniseis_pretrain.ipynb",
            "06_swinunetr_thebe_finetune_raw.ipynb",
            "07_swinunetr_thebe_finetune_clean_sr_cubes.ipynb",
            "08_omniseis_thebe_finetune_raw.ipynb",
            "09_omniseis_thebe_finetune_clean_cubes.ipynb",
            "10_omniseis_thebe_finetune_clean_sr_aug_reg.ipynb",
            "11_sr_training_seisgan.ipynb",
        }
        train_pattern = re.compile(r"(for\s+epoch\s+in|train_epoch\(|train_one_epoch\(|run_experiment\()")
        for path in notebooks:
            nb = json.loads(path.read_text(encoding="utf-8"))
            source = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
            self.assertIn("seismic_fault_recognition", source)
            self.assertIn("init_clearml_from_context", source)
            self.assertNotIn("source_" + "notebooks", source)
            self.assertNotIn("gdown", source)
            self.assertNotIn("drive.google", source)
            self.assertNotRegex(source, re.compile(r"^\s*class\s+\w+", re.MULTILINE), path.name)
            self.assertNotIn("NotImplementedError", source)
            if path.name in training_notebooks:
                self.assertRegex(source, train_pattern, path.name)
                self.assertIn("clearml_metric_logger", source)
            for cell in nb.get("cells", []):
                self.assertFalse(cell.get("outputs", []), path.name)


class DataTest(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy is not installed in this environment")
    def test_dataset_pairs_and_crops(self) -> None:
        from seismic_fault_recognition.data import SeisFaultDataset, analyze_npz

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            seis = path / "seis.npz"
            fault = path / "fault.npz"
            np.savez(seis, cube=np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8))
            np.savez(fault, cube=np.ones((8, 8, 8), dtype=np.float32))

            ds = SeisFaultDataset(seis, fault, target_shape=(4, 4, 4), random_crop=False)
            x, y = ds[0]
            self.assertEqual(tuple(x.shape[-3:]), (4, 4, 4))
            self.assertEqual(tuple(y.shape[-3:]), (4, 4, 4))
            ds.close()

            rows = analyze_npz(seis)
            self.assertEqual(rows[0]["shape"], (8, 8, 8))

    @unittest.skipIf(np is None, "numpy is not installed in this environment")
    def test_simmim_mask_shape(self) -> None:
        from seismic_fault_recognition.data import MaskGenerator3D

        mask = MaskGenerator3D(input_size=(8, 8, 8), mask_patch_size=4, mask_ratio=0.5, seed=1)()
        self.assertEqual(tuple(mask.shape[-3:]), (8, 8, 8))

    @unittest.skipIf(np is None, "numpy is not installed in this environment")
    def test_sr_degradation_shape(self) -> None:
        from seismic_fault_recognition.augmentations import build_sr_degradation_pipeline

        degrade = build_sr_degradation_pipeline(seed=1)
        cube = np.zeros((8, 8, 8), dtype=np.float32)
        self.assertEqual(degrade(cube).shape, cube.shape)

    @unittest.skipIf(np is None, "numpy is not installed in this environment")
    def test_normalize_zscore_accepts_read_only_arrays(self) -> None:
        from seismic_fault_recognition.data import normalize_zscore

        source = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        source.setflags(write=False)
        normalized = normalize_zscore(source)

        self.assertTrue(normalized.flags.writeable)
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=6)


class MetricsTest(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy is not installed in this environment")
    def test_binary_metrics(self) -> None:
        from seismic_fault_recognition.training import binary_metrics_np

        metrics = binary_metrics_np(np.array([0.1, 0.9, 0.8]), np.array([0, 1, 0]), threshold=0.5)
        self.assertGreater(metrics["recall"], 0.9)
        self.assertLess(metrics["precision"], 1.0)

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_segmentation_metrics_from_logits(self) -> None:
        from seismic_fault_recognition.metrics import segmentation_metrics_from_logits

        logits = torch.tensor([[[[[4.0, 3.0, 2.0, 1.0]]]]])
        target = torch.tensor([[[[[1.0, 0.0, 1.0, 0.0]]]]])
        metrics = segmentation_metrics_from_logits(logits, target, thresholds=(0.5, 0.9))

        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["iou"], 0.5)
        self.assertAlmostEqual(metrics["dice"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["ap"], 5.0 / 6.0, places=6)
        self.assertAlmostEqual(metrics["pr_auc"], 19.0 / 24.0, places=6)
        self.assertAlmostEqual(metrics["dice_best_threshold"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["f1@0.5"], 2.0 / 3.0)


class ProvenanceTest(unittest.TestCase):
    def test_data_and_experiment_manifest(self) -> None:
        from seismic_fault_recognition.provenance import build_data_manifest, save_experiment_manifest

        class Recipe:
            def as_dict(self) -> dict[str, str]:
                return {"name": "unit"}

        class Context:
            repo_root = Path(__file__).resolve().parents[1]
            recipe = Recipe()
            exp_config = {"stage": "unit"}
            base_config = None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.txt"
            sample.write_text("abc", encoding="utf-8")
            data_manifest = build_data_manifest({"sample": sample}, split_name="unit")
            self.assertTrue(data_manifest["files"][0]["exists"])
            self.assertEqual(data_manifest["files"][0]["size_bytes"], 3)

            out = save_experiment_manifest(Context(), data_manifest, root / "manifest.json")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["recipe"]["name"], "unit")
            self.assertIn("packages", payload)


class ClearMLTest(unittest.TestCase):
    def test_clearml_metric_decorator_logs_returned_metrics(self) -> None:
        from seismic_fault_recognition.clearml import ClearMLRun, clearml_metric_logger, report_optimizer_lr

        class FakeLogger:
            def __init__(self) -> None:
                self.scalars = []

            def report_scalar(self, **kwargs) -> None:
                self.scalars.append(kwargs)

        class FakeOptimizer:
            param_groups = [{"lr": 1e-4}]

        logger = FakeLogger()
        run = ClearMLRun(task=None, logger=logger, enabled=True, project_name="p", task_name="t")

        @clearml_metric_logger(run, series_prefix="Train")
        def one_epoch() -> dict[str, float]:
            return {"loss": 0.5, "f1": 0.75}

        self.assertEqual(one_epoch(clearml_iteration=3), {"loss": 0.5, "f1": 0.75})
        report_optimizer_lr(run, FakeOptimizer(), iteration=3)
        series = {item["series"] for item in logger.scalars}
        self.assertIn("Train/loss", series)
        self.assertIn("Train/f1", series)
        self.assertIn("LR", series)


class ModelImportTest(unittest.TestCase):
    def test_model_modules_import(self) -> None:
        import seismic_fault_recognition.models.faultformer  # noqa: F401
        import seismic_fault_recognition.models.omniseis  # noqa: F401
        import seismic_fault_recognition.models.swinunetr  # noqa: F401
        import seismic_fault_recognition.models.swinunetr_variants  # noqa: F401


class CLITest(unittest.TestCase):
    def run_cli(self, *args: str) -> tuple[int, str, str]:
        from seismic_fault_recognition.cli import main

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(args)
        # Combine them for some assertions that expect output in stdout
        combined = stdout.getvalue() + stderr.getvalue()
        return code, combined, stderr.getvalue()

    def test_recipes_list_and_show(self) -> None:
        code, stdout, stderr = self.run_cli("recipes", "list", "--json")
        self.assertEqual(code, 0, stderr)
        recipes = json.loads(stdout)
        self.assertEqual(len(recipes), 17)
        self.assertNotIn("source_" + "notebooks", stdout)

        code, stdout, stderr = self.run_cli("recipes", "show", "segmentation_validation", "--json")
        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["recipe"]["name"], "segmentation_validation")
        self.assertNotIn("source_" + "notebooks", stdout)

    def test_config_validate_cli(self) -> None:
        root = Path(__file__).resolve().parents[1]
        code, stdout, stderr = self.run_cli(
            "config",
            "validate",
            "--base",
            str(root / "configs" / "datasphere.yaml"),
            "--experiments",
            str(root / "configs" / "experiments"),
        )
        self.assertEqual(code, 0, stderr)
        self.assertIn("OK:", stdout)

    @unittest.skipIf(np is None, "numpy is not installed in this environment")
    def test_data_audit_cli_on_tiny_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                np.savez(root / f"{split}_seis.npz", **{split: np.zeros((2, 2, 2), dtype=np.float32)})
                np.savez(root / f"{split}_fault.npz", **{split: np.zeros((2, 2, 2), dtype=np.float32)})
            experiment = root / "experiment.yaml"
            experiment.write_text(_minimal_experiment_yaml(), encoding="utf-8")
            output = root / "audit.json"

            code, stdout, stderr = self.run_cli(
                "data",
                "audit",
                "--experiment",
                str(experiment),
                "--output",
                str(output),
                "--base-dir",
                str(root),
            )
            self.assertEqual(code, 0, stderr)
            self.assertIn("status: ok", stdout)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ok")

    @unittest.skipIf(np is None or torch is None, "NumPy and PyTorch are required for checkpoint CLI")
    def test_checkpoint_inspect_cli(self) -> None:
        from seismic_fault_recognition.training import save_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pth"
            model = torch.nn.Conv3d(1, 1, kernel_size=1)
            save_checkpoint(path, model, epoch=1, metrics={"val_dice_best_threshold": 1.0})

            code, stdout, stderr = self.run_cli("checkpoint", "inspect", str(path), "--json", "--no-block-stats")
            self.assertEqual(code, 0, stderr)
            payload = json.loads(stdout)
            self.assertEqual(payload["epoch"], 1)
            self.assertIn("swinunetr", payload)

    @unittest.skipIf(np is None or torch is None, "NumPy and PyTorch are required for validation CLI")
    def test_segmentation_validate_cli_tiny_cpu(self) -> None:
        from seismic_fault_recognition.models.factory import build_model_by_name
        from seismic_fault_recognition.training import save_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "test_seis.npz", test=np.zeros((16, 16, 16), dtype=np.float32))
            np.savez(root / "test_fault.npz", test=np.zeros((16, 16, 16), dtype=np.float32))
            for split in ("train", "val"):
                np.savez(root / f"{split}_seis.npz", **{split: np.zeros((2, 2, 2), dtype=np.float32)})
                np.savez(root / f"{split}_fault.npz", **{split: np.zeros((2, 2, 2), dtype=np.float32)})
            experiment = root / "experiment.yaml"
            experiment.write_text(_minimal_experiment_yaml(model_variant="faultformer_attention", roi_size="[16, 16, 16]"), encoding="utf-8")
            checkpoint = root / "checkpoint.pth"
            model = build_model_by_name("faultformer_attention")
            save_checkpoint(checkpoint, model, epoch=1, metrics={})
            output = root / "metrics.json"

            code, _stdout, stderr = self.run_cli(
                "validate",
                "segmentation",
                "--experiment",
                str(experiment),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--base-dir",
                str(root),
                "--base-config",
                str(Path(__file__).resolve().parents[1] / "configs" / "datasphere.yaml"),
                "--device",
                "cpu",
            )
            self.assertEqual(code, 0, stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("val_dice_best_threshold", payload)


class RepositoryHygieneTest(unittest.TestCase):
    def test_no_legacy_notebook_lineage(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "notebooks" / ("_" + "archive")).exists())
        targets = [
            root / "README.md",
            *(root / "docs").glob("*.md"),
            *root.joinpath("src").rglob("*.py"),
            *root.joinpath("notebooks").glob("*.ipynb"),
        ]
        forbidden = (
            "source_" + "notebooks",
            "notebooks/" + "_" + "archive",
            "archive" + "/raw",
            "Source " + "notebook",
            "Source " + "notebooks",
            "recovered " + "from",
            "older " + "notebooks",
            "old " + "notebooks",
        )
        for path in targets:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, str(path))


class TrainerIntegrationTest(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_masked_simmim_loss_uses_only_masked_voxels(self) -> None:
        from seismic_fault_recognition.trainers import masked_simmim_l1_loss

        pred = torch.tensor([[[[[1.0, 3.0]]]]])
        target = torch.zeros_like(pred)
        mask = torch.tensor([[[[[1.0, 0.0]]]]])
        self.assertAlmostEqual(float(masked_simmim_l1_loss(pred, target, mask)), 1.0)

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_accumulation_steps_flushes_tail_batch(self) -> None:
        from seismic_fault_recognition.trainers import train_thebe_finetune_epoch

        class CountingSGD(torch.optim.SGD):
            def __init__(self, params) -> None:
                super().__init__(params, lr=0.1)
                self.step_count = 0

            def step(self, closure=None):  # type: ignore[override]
                self.step_count += 1
                return super().step(closure)

        model = torch.nn.Conv3d(1, 1, kernel_size=1)
        optimizer = CountingSGD(model.parameters())
        loss_fn = torch.nn.BCEWithLogitsLoss()
        batch = (torch.zeros(1, 1, 2, 2, 2), torch.zeros(1, 1, 2, 2, 2))
        train_thebe_finetune_epoch(model, [batch, batch, batch], optimizer, loss_fn, device="cpu", amp=False, accumulation_steps=2)
        self.assertEqual(optimizer.step_count, 2)

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_tiny_cpu_train_checkpoint_validate_integration(self) -> None:
        from seismic_fault_recognition.trainers import train_thebe_finetune_epoch, validate_thebe_finetune
        from seismic_fault_recognition.training import save_checkpoint

        model = torch.nn.Conv3d(1, 1, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        batch = (
            torch.zeros(1, 1, 2, 2, 2),
            torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 0.0]]]]]),
        )

        with tempfile.TemporaryDirectory() as tmp:
            loss = train_thebe_finetune_epoch(model, [batch], optimizer, loss_fn, device="cpu", amp=False)
            metrics = validate_thebe_finetune(model, [batch], loss_fn, device="cpu", thresholds=(0.5, 0.7))
            checkpoint_path = Path(tmp) / "tiny.pth"
            save_checkpoint(checkpoint_path, model, optimizer=optimizer, epoch=1, metrics=metrics)

            self.assertTrue(checkpoint_path.exists())
            self.assertGreaterEqual(loss, 0.0)
            self.assertIn("dice_best_threshold", metrics)
            self.assertIn("val_dice_best_threshold", metrics)
            self.assertIn("per_threshold", metrics)

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_simmim_three_tensor_batch_trains_and_validates(self) -> None:
        from seismic_fault_recognition.trainers import train_simmim_epoch, validate_simmim_reconstruction

        model = torch.nn.Conv3d(1, 1, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.L1Loss()
        target = torch.ones(1, 1, 4, 4, 4)
        mask = torch.zeros_like(target)
        mask[..., :2, :, :] = 1.0
        masked = target * (1.0 - mask)
        loader = [(masked, target, mask)]

        train_loss = train_simmim_epoch(model, loader, optimizer, loss_fn, device="cpu", amp=False)
        metrics = validate_simmim_reconstruction(model, loader, loss_fn, device="cpu")

        self.assertGreaterEqual(train_loss, 0.0)
        self.assertEqual(metrics["loss"], metrics["val_loss"])

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_faultseg_tuple_batch_trains_and_returns_threshold_metrics(self) -> None:
        from seismic_fault_recognition.trainers import evaluate_faultseg3d, train_faultseg3d_epoch

        model = torch.nn.Conv3d(1, 1, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        batch = (torch.zeros(1, 1, 4, 4, 4), torch.zeros(1, 1, 4, 4, 4))
        loader = [batch]

        train_loss = train_faultseg3d_epoch(model, loader, optimizer, loss_fn, device="cpu", amp=False)
        metrics = evaluate_faultseg3d(
            model,
            loader,
            loss_fn,
            device="cpu",
            threshold=0.5,
            thresholds=(0.3, 0.5),
        )

        self.assertGreaterEqual(train_loss, 0.0)
        self.assertIn("val_dice_best_threshold", metrics)
        self.assertIn("per_threshold", metrics)

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_sr_notebook_call_contract_trains_and_validates(self) -> None:
        from seismic_fault_recognition.losses import build_loss
        from seismic_fault_recognition.trainers import train_sr_epoch, validate_sr

        generator = torch.nn.Conv3d(1, 1, kernel_size=1)
        discriminator = torch.nn.Conv3d(1, 1, kernel_size=1)
        optimizer_g = torch.optim.SGD(generator.parameters(), lr=0.1)
        optimizer_d = torch.optim.SGD(discriminator.parameters(), lr=0.1)
        loss_fn = build_loss("sr_l1_vgg_gan", use_vgg=False)
        loader = [(torch.zeros(1, 1, 4, 4, 4), torch.ones(1, 1, 4, 4, 4))]

        train_metrics = train_sr_epoch(
            generator,
            loader,
            optimizer_g,
            loss_fn,
            device="cpu",
            amp=False,
            discriminator=discriminator,
            optimizer_d=optimizer_d,
            return_metrics=True,
            metric_normalization="fixed_range",
            metric_data_range=(-1.0, 1.0),
        )
        val_metrics = validate_sr(
            generator,
            loader,
            loss_fn,
            device="cpu",
            metric_normalization="fixed_range",
            metric_data_range=(-1.0, 1.0),
        )

        self.assertIn("loss_g", train_metrics)
        self.assertIn("psnr", train_metrics)
        self.assertIn("psnr", val_metrics)
        self.assertIn("val_psnr", val_metrics)


def _minimal_experiment_yaml(model_variant: str = "faultformer_attention", roi_size: str = "[2, 2, 2]") -> str:
    return f"""recipe: segmentation_validation
notebook: 12_segmentation_validation.ipynb
stage: validation
dataset: test
model_variant: {model_variant}
loss_profile: segmentation_bce
augmentation_profile: thebe_random_crop
trainer_profile: segmentation_validation
data:
  train_seis: train_seis.npz
  train_fault: train_fault.npz
  val_seis: val_seis.npz
  val_fault: val_fault.npz
  test_seis: test_seis.npz
  test_fault: test_fault.npz
checkpoints:
  output_dir: checkpoints
  best: checkpoints/best.pth
  latest: checkpoints/latest.pth
outputs:
  output_dir: outputs
training:
  roi_size: {roi_size}
  patch_size: {roi_size}
  batch_size: 1
  learning_rate: 0.0001
  max_epochs: 1
  threshold: 0.5
  overlap: 0.5
  num_workers: 0
  seed: 42
"""


if __name__ == "__main__":
    unittest.main()
