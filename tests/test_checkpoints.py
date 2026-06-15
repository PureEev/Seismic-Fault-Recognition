from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
except ImportError:  # pragma: no cover - local minimal Python image.
    np = None  # type: ignore


class FakeLoadResult:
    missing_keys = ("missing.weight",)
    unexpected_keys = ("unexpected.weight",)


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def load_state_dict(self, state_dict, strict: bool = True):
        self.calls.append(strict)
        if strict:
            raise RuntimeError("strict load failed")
        self.state_dict = state_dict
        return FakeLoadResult()


@unittest.skipIf(np is None, "numpy is not installed in this environment")
class CheckpointDiagnosticsTest(unittest.TestCase):
    def test_clean_state_dict_strips_module_prefix(self) -> None:
        from seismic_fault_recognition.checkpoints import clean_state_dict

        cleaned = clean_state_dict({"module.encoder1.weight": np.ones(1), "plain": np.zeros(1)})
        self.assertIn("encoder1.weight", cleaned)
        self.assertIn("plain", cleaned)

    def test_load_state_dict_loose_falls_back_to_non_strict(self) -> None:
        from seismic_fault_recognition.checkpoints import load_state_dict_loose

        model = FakeModel()
        result = load_state_dict_loose(model, {"state_dict": {"module.encoder1.weight": np.ones(1)}})

        self.assertTrue(result.success)
        self.assertFalse(result.strict)
        self.assertEqual(model.calls, [True, False])
        self.assertIn("encoder1.weight", model.state_dict)
        self.assertEqual(result.missing_keys, ("missing.weight",))
        self.assertEqual(result.unexpected_keys, ("unexpected.weight",))

    def test_inspect_swinunetr_checkpoint_detects_variant_features(self) -> None:
        from seismic_fault_recognition.checkpoints import inspect_swinunetr_checkpoint

        info = inspect_swinunetr_checkpoint(
            {
                "state_dict": {
                    "module.swinViT.patch_embed.proj.weight": np.zeros((48, 1, 5, 5, 5), dtype=np.float32),
                    "module.enc2to3.weight": np.zeros((192, 96, 1, 1, 1), dtype=np.float32),
                    "module.out.conv.conv.weight": np.zeros((1, 24, 1, 1, 1), dtype=np.float32),
                }
            }
        )

        self.assertTrue(info.has_patch_embed)
        self.assertTrue(info.has_patch5)
        self.assertTrue(info.has_adapters)
        self.assertFalse(info.has_layer4)
        self.assertEqual(info.in_channels, 1)
        self.assertEqual(info.feature_size, 48)
        self.assertEqual(info.out_channels, 1)

    def test_summarize_block_weights_is_deterministic(self) -> None:
        from seismic_fault_recognition.checkpoints import summarize_block_weights

        stats = summarize_block_weights(
            {
                "encoder1.weight": np.array([3.0, 4.0], dtype=np.float32),
                "encoder1.bias": np.array([100.0], dtype=np.float32),
                "decoder1.weight": np.array([0.0, 0.0005], dtype=np.float32),
            },
            blocks=("encoder1", "decoder1"),
        )

        by_block = {item.block: item for item in stats}
        self.assertAlmostEqual(by_block["encoder1"].l2_norm, 5.0)
        self.assertEqual(by_block["encoder1"].num_weights, 2)
        self.assertAlmostEqual(by_block["encoder1"].sparsity_pct, 0.0)
        self.assertAlmostEqual(by_block["decoder1"].sparsity_pct, 100.0)


if __name__ == "__main__":
    unittest.main()
