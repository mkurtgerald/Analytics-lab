import hashlib
import unittest
from unittest import mock

from analytics_lab import lpr_ocr_exact_evidence as evidence


class ExactOCREvidenceTests(unittest.TestCase):
    def test_git_blob_identity_matches_git_object_rule(self):
        payload = b"abc"
        expected = hashlib.sha1(b"blob 3\0abc").hexdigest()
        self.assertEqual(evidence._git_blob_sha1(payload), expected)

    def test_sha256_verification_is_fail_closed(self):
        payload = b"bounded"
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(
            evidence._verify_sha256(payload, size=len(payload), sha256=digest, name="fixture"),
            digest,
        )
        with self.assertRaises(RuntimeError):
            evidence._verify_sha256(payload, size=len(payload) + 1, sha256=digest, name="fixture")
        with self.assertRaises(RuntimeError):
            evidence._verify_sha256(payload, size=len(payload), sha256="0" * 64, name="fixture")

    def test_ppm_requires_exact_rgb24_length(self):
        self.assertEqual(
            evidence._ppm(b"\x01\x02\x03", width=1, height=1),
            b"P6\n1 1\n255\n\x01\x02\x03",
        )
        with self.assertRaises(ValueError):
            evidence._ppm(b"\x01\x02", width=1, height=1)

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_work_dir_requires_runner_temp(self):
        with self.assertRaises(RuntimeError):
            evidence._bounded_work_dir(evidence.Path("child"))


if __name__ == "__main__":
    unittest.main()
