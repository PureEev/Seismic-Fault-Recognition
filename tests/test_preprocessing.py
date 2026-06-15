from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
except ImportError:  # pragma: no cover - local minimal Python image.
    np = None  # type: ignore


@unittest.skipIf(np is None, "numpy is not installed in this environment")
class SeisGAN2DPreprocessingTest(unittest.TestCase):
    def test_inline_crossline_slice_count(self) -> None:
        from seismic_fault_recognition.preprocessing.seisgan2d import extract_slice_records

        cube = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
        records = extract_slice_records(cube, patch=(2, 2), slice_stride=2, stride=2)

        self.assertEqual(len(records), 16)
        self.assertEqual({record.orientation for record in records}, {"inline", "crossline"})
        self.assertEqual(records[0].array.shape, (2, 2))

    def test_degradation_pipeline_shapes_and_values(self) -> None:
        from seismic_fault_recognition.preprocessing.seisgan2d import (
            add_noise_by_snr,
            is_empty_slice,
            lowpass_time,
            make_lr_hr_pairs,
            normalize_minmax,
            tsp_downsample,
        )

        patch = np.arange(16, dtype=np.float32).reshape(4, 4)
        self.assertFalse(is_empty_slice(patch))
        self.assertTrue(is_empty_slice(np.zeros((4, 4), dtype=np.float32)))

        normalized = normalize_minmax(patch)
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

        filtered = lowpass_time(patch, f0=15, dt=0.002)
        self.assertEqual(filtered.shape, patch.shape)
        self.assertTrue(np.isfinite(filtered).all())

        noisy = add_noise_by_snr(filtered, snr_db=20, seed=1)
        self.assertEqual(noisy.shape, patch.shape)
        self.assertTrue(np.isfinite(noisy).all())

        downsampled = tsp_downsample(patch)
        self.assertEqual(downsampled.shape, (2, 2))

        pairs = make_lr_hr_pairs([patch], seed=1)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0].shape, (2, 2))
        self.assertEqual(pairs[0][1].shape, (4, 4))


@unittest.skipIf(np is None, "numpy is not installed in this environment")
class ThebeCropPreprocessingTest(unittest.TestCase):
    def test_bounds_grid_and_crop_saving(self) -> None:
        from seismic_fault_recognition.preprocessing.thebe_crops import crop_grid, find_valid_bounds, save_valid_crops

        seis = np.zeros((4, 4, 4), dtype=np.float32)
        seis[1:4, 1:4, 1:4] = np.arange(1, 28, dtype=np.float32).reshape(3, 3, 3)
        fault = np.zeros_like(seis, dtype=np.int8)
        fault[1:4, 1:4, 1:4] = 1

        bounds, footprint = find_valid_bounds(seis, chunk_size=2)
        self.assertIsNotNone(bounds)
        assert bounds is not None
        self.assertEqual(bounds.as_dict()["x_min"], 1)
        self.assertEqual(tuple(footprint.shape), (4, 4))

        self.assertEqual(len(crop_grid((4, 4, 4), crop_size=(2, 2, 2), overlap=0.0)), 8)

        with tempfile.TemporaryDirectory() as tmp:
            report = save_valid_crops(
                seis,
                fault,
                tmp,
                crop_size=(2, 2, 2),
                overlap=0.0,
                prefix="unit",
                min_footprint=1.0,
                chunk_size=2,
            )
            self.assertGreater(report.saved_count, 0)
            first = report.records[0]
            self.assertTrue(Path(first.seis_path).exists())
            self.assertTrue(Path(first.fault_path).exists())

    def test_assemble_npz_chunks_to_memmap(self) -> None:
        from seismic_fault_recognition.preprocessing.thebe_crops import assemble_npz_chunks_to_memmap

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "chunk1.npz", arr_0=np.ones((1, 2, 2), dtype=np.float32))
            np.savez(root / "chunk2.npz", arr_0=np.full((2, 2, 2), 2, dtype=np.float32))
            mmap = assemble_npz_chunks_to_memmap(root, "chunk", root / "merged.npy", num_chunks=2, axis=0)

            self.assertEqual(tuple(mmap.shape), (3, 2, 2))
            self.assertAlmostEqual(float(mmap[0].mean()), 1.0)
            self.assertAlmostEqual(float(mmap[1:].mean()), 2.0)


if __name__ == "__main__":
    unittest.main()
