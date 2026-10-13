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

    def test_manifest_has_one_positive_and_floor_transition_hard_negative(self):
        identities = {item.sample_id: ("a" if index == 0 else "b") * 64 for index, item in enumerate(GMDCSA24_SEED)}
        document = _manifest(identities)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["config"]["max_samples"], 2)
        self.assertEqual(document["config"]["max_total_video_bytes"], 9_347_373)
        self.assertEqual(len(document["samples"][0]["labels"]), 1)
        self.assertEqual(document["samples"][1]["labels"], [])
        self.assertEqual(document["samples"][0]["end_timestamp_ms"], 8000)
        self.assertEqual(document["samples"][1]["end_timestamp_ms"], 7000)
        self.assertEqual(document["samples"][0]["labels"][0]["start_timestamp_ms"], 2500)
        self.assertEqual(document["samples"][0]["labels"][0]["end_timestamp_ms"], 8000)
        self.assertEqual(document["samples"][0]["sample_id"], "gmdcsa24-s2-fall-13")
        self.assertEqual(document["samples"][1]["sample_id"], "gmdcsa24-s2-adl-10")
        self.assertEqual(document["samples"][0]["site_id"], document["samples"][1]["site_id"])
        self.assertNotEqual(document["samples"][0]["camera_id"], document["samples"][1]["camera_id"])
        self.assertTrue(all(sample["camera_id"].startswith("gmdcsa24-clip-") for sample in document["samples"]))

    def test_attribution_tracks_exact_active_files(self):
        text = _attribution_text()
        self.assertIn("Subject 2/Fall/13.mp4", text)
        self.assertIn("Subject 2/ADL/10.mp4", text)
        self.assertNotIn("Subject 1/Fall/11.mp4", text)
        self.assertNotIn("Subject 4/Fall/03.mp4", text)
        self.assertNotIn("Subject 3/Fall/03.mp4", text)
        self.assertNotIn("Subject 2/Fall/01.mp4", text)
        self.assertNotIn("Subject 1/Fall/05.mp4", text)
        self.assertIn("git:5abac7693229900cf80f722e878fbb119211fc1c", text)
        self.assertIn("MIT", text)

    def test_subject4_regression_selection_binds_manifest_labels_budget_and_attribution(self):
        identities = {item.sample_id: ("a" if index == 0 else "b") * 64 for index, item in enumerate(GMDCSA24_HELDOUT_S4)}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(validation_seed, "_download_media", return_value=identities) as media, \
                patch.object(validation_seed, "_download_models") as models:
            opener = object()  # Acquisition is stubbed; no network/model/media execution.
            manifest = prepare_seed(temp, subset="subject4-regression", opener=opener)
            media.assert_called_once_with(Path(temp).resolve(), opener=opener, specs=GMDCSA24_HELDOUT_S4)
            models.assert_called_once_with(Path(temp).resolve(), opener=opener)
            document = json.loads(manifest.read_text())
            _, samples, config = load_manifest(manifest)
            attribution = (Path(temp) / "ATTRIBUTION.txt").read_text()
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(config.max_samples, 2)
        self.assertEqual(config.max_total_video_bytes, 7_996_782)
        self.assertLess(config.max_total_video_bytes, 16 * 1024 * 1024)
        self.assertEqual([sample.sample_id for sample in samples], [item.sample_id for item in GMDCSA24_HELDOUT_S4])
        self.assertEqual(samples[0].end_timestamp_ms, 6000)
        self.assertEqual(samples[1].end_timestamp_ms, 4000)
        self.assertEqual((samples[0].labels[0].start_timestamp_ms, samples[0].labels[0].end_timestamp_ms), (1700, 6000))
        self.assertEqual(samples[1].labels, ())
        self.assertEqual([sample.media_sha256 for sample in samples], ["a" * 64, "b" * 64])
        self.assertIn("Subject 4/Fall/03.mp4", attribution)
        self.assertIn("Subject 4/ADL/07.mp4", attribution)
        self.assertNotIn("Subject 2/Fall/13.mp4", attribution)
        self.assertIn("MIT", attribution)
        self.assertEqual(GMDCSA24_SEED, GMDCSA24_ROBUSTNESS_S2_NIGHT)

    def test_prepare_default_still_routes_active_seed(self):
        identities = {item.sample_id: "a" * 64 for item in GMDCSA24_SEED}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(validation_seed, "_download_media", return_value=identities) as media, \
                patch.object(validation_seed, "_download_models"):
            opener = object()
            manifest = prepare_seed(temp, opener=opener)
            media.assert_called_once_with(Path(temp).resolve(), opener=opener, specs=GMDCSA24_SEED)
            self.assertEqual(json.loads(manifest.read_text())["config"]["max_total_video_bytes"], 9_347_373)

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

    def test_subject4_download_routes_only_exact_reviewed_media_specs(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(validation_seed, "_download_exact") as download, \
                patch.object(validation_seed, "verify_seed_media", side_effect=("a" * 64, "b" * 64)) as verify:
            opener = object()
            root = Path(temp).resolve()
            identities = _download_media(root, opener=opener, specs=GMDCSA24_HELDOUT_S4)
            self.assertEqual(download.call_count, 2)
            self.assertEqual(verify.call_count, 2)
            for index, spec in enumerate(GMDCSA24_HELDOUT_S4):
                target = root / "media" / "gmdcsa24" / spec.relative_path
                self.assertEqual(download.call_args_list[index].args, (spec.source_url, target))
                self.assertEqual(download.call_args_list[index].kwargs, {"expected_size": spec.size_bytes, "opener": opener})
                self.assertEqual(verify.call_args_list[index].args, (target, spec))
                self.assertEqual(identities[spec.sample_id], ("a" if index == 0 else "b") * 64)


if __name__ == "__main__":
    unittest.main()
