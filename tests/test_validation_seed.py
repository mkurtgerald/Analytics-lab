import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analytics_lab import validation_seed
from analytics_lab.validation_cli import load_manifest
from analytics_lab.validation_seed import (
    GMDCSA24_ACCEPTED_SEED,
    GMDCSA24_HELDOUT_S2,
    GMDCSA24_HELDOUT_S3,
    GMDCSA24_HELDOUT_S4,
    GMDCSA24_ROBUSTNESS_S1_BW,
    GMDCSA24_ROBUSTNESS_S2_NIGHT,
    GMDCSA24_SEED,
    SeedMediaSpec,
    _attribution_text,
    _download_exact,
    _download_media,
    _manifest,
    prepare_seed,
    verify_seed_media,
)


class _Response(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()
        return False


class ValidationSeedTests(unittest.TestCase):
    def test_active_subset_is_untouched_file_pair_under_bounded_budget(self):
        self.assertEqual(GMDCSA24_SEED, GMDCSA24_ROBUSTNESS_S2_NIGHT)
        self.assertEqual(len(GMDCSA24_ACCEPTED_SEED), 2)
        self.assertEqual(len(GMDCSA24_HELDOUT_S2), 2)
        self.assertEqual(len(GMDCSA24_HELDOUT_S3), 2)
        self.assertEqual(len(GMDCSA24_HELDOUT_S4), 2)
        self.assertEqual(len(GMDCSA24_ROBUSTNESS_S1_BW), 2)
        self.assertEqual(len(GMDCSA24_ROBUSTNESS_S2_NIGHT), 2)
        self.assertLess(sum(item.size_bytes for item in GMDCSA24_SEED), 16 * 1024 * 1024)
        self.assertEqual(sum(item.label_id is not None for item in GMDCSA24_SEED), 1)
        self.assertEqual(sum(item.label_id is None for item in GMDCSA24_SEED), 1)
        self.assertTrue(all(item.relative_path.startswith("Subject 2/") for item in GMDCSA24_SEED))
        historical_paths = {
            item.relative_path
            for seed in (
                GMDCSA24_ACCEPTED_SEED,
                GMDCSA24_HELDOUT_S2,
                GMDCSA24_HELDOUT_S3,
                GMDCSA24_HELDOUT_S4,
                GMDCSA24_ROBUSTNESS_S1_BW,
            )
            for item in seed
        }
        self.assertTrue(historical_paths.isdisjoint(item.relative_path for item in GMDCSA24_SEED))
        self.assertIn("%20", GMDCSA24_SEED[0].source_url)

    def test_verify_seed_media_binds_git_blob_and_sha256(self):
        payload = b"bounded-real-video-fixture"
        git_sha1 = hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
        spec = SeedMediaSpec("sample", "x.mp4", len(payload), git_sha1, 1000)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "x.mp4"
            path.write_bytes(payload)
            self.assertEqual(verify_seed_media(path, spec), hashlib.sha256(payload).hexdigest())
            path.write_bytes(payload + b"x")
            with self.assertRaises(ValueError):
                verify_seed_media(path, spec)

    def test_download_exact_is_bounded_and_does_not_leave_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "x.bin"
            def opener(_request, timeout):
                self.assertEqual(timeout, 30)
                return _Response(b"abcd")
            with self.assertRaises(RuntimeError):
                _download_exact("https://example.com/x", target, expected_size=3, opener=opener)
            self.assertFalse(target.exists())
            self.assertFalse((Path(temp) / "x.bin.partial").exists())

    def test_manifest_rejects_held_media_rights(self):
        identities = {item.sample_id: "a" * 64 for item in GMDCSA24_SEED}
        with self.assertRaisesRegex(ValueError, "not eligible"):
            _manifest(identities)
        self.assertEqual([item.sample_id for item in GMDCSA24_SEED],
                         ["gmdcsa24-s2-fall-13", "gmdcsa24-s2-adl-10"])

    def test_attribution_rejects_held_media_rights(self):
        with self.assertRaisesRegex(ValueError, "not eligible"):
            _attribution_text()
        self.assertEqual([item.relative_path for item in GMDCSA24_SEED],
                         ["Subject 2/Fall/13.mp4", "Subject 2/ADL/10.mp4"])

    def test_subject4_regression_denies_before_directory_or_acquisition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "held"
            with patch.object(validation_seed, "_download_media") as media, \
                    patch.object(validation_seed, "_download_models") as models:
                with self.assertRaisesRegex(ValueError, "not eligible"):
                    prepare_seed(root, subset="subject4-regression", opener=object())
                self.assertFalse(root.exists())
                media.assert_not_called()
                models.assert_not_called()
        self.assertEqual([item.relative_path for item in GMDCSA24_HELDOUT_S4],
                         ["Subject 4/Fall/03.mp4", "Subject 4/ADL/07.mp4"])

    def test_prepare_default_denies_before_directory_or_acquisition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "held"
            with patch.object(validation_seed, "_download_media") as media, \
                    patch.object(validation_seed, "_download_models") as models:
                with self.assertRaisesRegex(ValueError, "not eligible"):
                    prepare_seed(root, opener=object())
                self.assertFalse(root.exists())
                media.assert_not_called()
                models.assert_not_called()

    def test_unknown_subset_rejected_before_directory_or_asset_acquisition(self):
        with patch.object(validation_seed, "_ensure_root") as root, \
                patch.object(validation_seed, "_download_media") as media, \
                patch.object(validation_seed, "_download_models") as models:
            for subset in ("subject4", "Subject4-regression", "../subject4-regression", "", None, False):
                with self.subTest(subset=subset), self.assertRaises(ValueError):
                    prepare_seed("unused", subset=subset)
            root.assert_not_called()
            media.assert_not_called()
            models.assert_not_called()

    def test_direct_media_helper_denies_before_directory_or_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "held"
            with patch.object(validation_seed, "_download_exact") as network:
                with self.assertRaisesRegex(ValueError, "not eligible"):
                    _download_media(root, opener=object(), specs=GMDCSA24_HELDOUT_S4)
                self.assertFalse(root.exists())
                network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
