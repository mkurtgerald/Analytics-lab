import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from analytics_lab.validation_seed import (
    GMDCSA24_ACCEPTED_SEED,
    GMDCSA24_HELDOUT_S2,
    GMDCSA24_HELDOUT_S3,
    GMDCSA24_HELDOUT_S4,
    GMDCSA24_ROBUSTNESS_S1_BW,
    GMDCSA24_ROBUSTNESS_S2_NIGHT,
    GMDCSA24_ROBUSTNESS_S3_NIGHT_SW,
    GMDCSA24_SEED,
    SeedMediaSpec,
    _attribution_text,
    _download_exact,
    _manifest,
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
        self.assertEqual(GMDCSA24_SEED, GMDCSA24_ROBUSTNESS_S3_NIGHT_SW)
        self.assertEqual(len(GMDCSA24_ACCEPTED_SEED), 2)
        self.assertEqual(len(GMDCSA24_HELDOUT_S2), 2)
        self.assertEqual(len(GMDCSA24_HELDOUT_S3), 2)
        self.assertEqual(len(GMDCSA24_HELDOUT_S4), 2)
        self.assertEqual(len(GMDCSA24_ROBUSTNESS_S1_BW), 2)
        self.assertEqual(len(GMDCSA24_ROBUSTNESS_S2_NIGHT), 2)
        self.assertEqual(len(GMDCSA24_ROBUSTNESS_S3_NIGHT_SW), 2)
        self.assertLess(sum(item.size_bytes for item in GMDCSA24_SEED), 16 * 1024 * 1024)
        self.assertEqual(sum(item.label_id is not None for item in GMDCSA24_SEED), 1)
        self.assertEqual(sum(item.label_id is None for item in GMDCSA24_SEED), 1)
        self.assertTrue(all(item.relative_path.startswith("Subject 3/") for item in GMDCSA24_SEED))
        historical_paths = {
            item.relative_path
            for seed in (
                GMDCSA24_ACCEPTED_SEED,
                GMDCSA24_HELDOUT_S2,
                GMDCSA24_HELDOUT_S3,
                GMDCSA24_HELDOUT_S4,
                GMDCSA24_ROBUSTNESS_S1_BW,
                GMDCSA24_ROBUSTNESS_S2_NIGHT,
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

    def test_manifest_has_one_positive_and_exercise_floor_transition_hard_negative(self):
        identities = {item.sample_id: ("a" if index == 0 else "b") * 64 for index, item in enumerate(GMDCSA24_SEED)}
        document = _manifest(identities)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["config"]["max_samples"], 2)
        self.assertEqual(document["config"]["max_total_video_bytes"], 12_434_507)
        self.assertEqual(len(document["samples"][0]["labels"]), 1)
        self.assertEqual(document["samples"][1]["labels"], [])
        self.assertEqual(document["samples"][0]["end_timestamp_ms"], 7000)
        self.assertEqual(document["samples"][1]["end_timestamp_ms"], 8000)
        self.assertEqual(document["samples"][0]["labels"][0]["start_timestamp_ms"], 1800)
        self.assertEqual(document["samples"][0]["labels"][0]["end_timestamp_ms"], 6500)
        self.assertEqual(document["samples"][0]["sample_id"], "gmdcsa24-s3-fall-16")
        self.assertEqual(document["samples"][1]["sample_id"], "gmdcsa24-s3-adl-06")
        self.assertEqual(document["samples"][0]["site_id"], document["samples"][1]["site_id"])
        self.assertNotEqual(document["samples"][0]["camera_id"], document["samples"][1]["camera_id"])
        self.assertTrue(all(sample["camera_id"].startswith("gmdcsa24-clip-") for sample in document["samples"]))

    def test_attribution_tracks_exact_active_files(self):
        text = _attribution_text()
        self.assertIn("Subject 3/Fall/16.mp4", text)
        self.assertIn("Subject 3/ADL/06.mp4", text)
        self.assertNotIn("Subject 2/Fall/13.mp4", text)
        self.assertNotIn("Subject 2/ADL/10.mp4", text)
        self.assertNotIn("Subject 1/Fall/11.mp4", text)
        self.assertNotIn("Subject 4/Fall/03.mp4", text)
        self.assertNotIn("Subject 3/Fall/03.mp4", text)
        self.assertNotIn("Subject 2/Fall/01.mp4", text)
        self.assertNotIn("Subject 1/Fall/05.mp4", text)
        self.assertIn("git:5abac7693229900cf80f722e878fbb119211fc1c", text)
        self.assertIn("MIT", text)


if __name__ == "__main__":
    unittest.main()