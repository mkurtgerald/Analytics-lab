import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from analytics_lab.validation_seed import (
    GMDCSA24_SEED,
    SeedMediaSpec,
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
    def test_pinned_seed_is_two_real_world_clips_under_bounded_budget(self):
        self.assertEqual(len(GMDCSA24_SEED), 2)
        self.assertLess(sum(item.size_bytes for item in GMDCSA24_SEED), 16 * 1024 * 1024)
        self.assertIsNotNone(GMDCSA24_SEED[0].label_id)
        self.assertIsNone(GMDCSA24_SEED[1].label_id)
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

    def test_manifest_has_positive_and_hard_negative_with_exact_identities(self):
        identities = {item.sample_id: ("a" if index == 0 else "b") * 64 for index, item in enumerate(GMDCSA24_SEED)}
        document = _manifest(identities)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["config"]["max_samples"], 2)
        self.assertEqual(len(document["samples"][0]["labels"]), 1)
        self.assertEqual(document["samples"][1]["labels"], [])
        self.assertEqual(document["samples"][0]["end_timestamp_ms"], 5000)
        self.assertEqual(document["samples"][1]["end_timestamp_ms"], 7000)


if __name__ == "__main__":
    unittest.main()
