import hashlib
import subprocess
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

    def test_preprocess_is_preregistered_by_exact_branch_family(self):
        self.assertEqual(
            evidence._selected_preprocess("evidence/lpr-ocr-exact-preprocess-1"),
            "gray-otsu",
        )
        self.assertEqual(evidence._selected_preprocess("evidence/lpr-ocr-exact-1"), "raw")
        self.assertEqual(evidence._selected_preprocess("feature/unrelated"), "raw")
        self.assertEqual(evidence._selected_preprocess(""), "raw")
        with self.assertRaises(ValueError):
            evidence._selected_preprocess(None)

    def test_official_windows_package_version_is_exact_and_canonicalized(self):
        def fake_runner(args, **kwargs):
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=b"tesseract v5.5.3.20260724\nleptonica-1.87.0\n",
                stderr=b"",
            )

        runner = evidence._OfficialWindowsPackageRunner("tesseract.exe", runner=fake_runner)
        result = runner(["tesseract.exe", "--version"], capture_output=True, check=False)
        self.assertEqual(runner.reported_version, "tesseract v5.5.3.20260724")
        self.assertEqual(result.stdout, b"tesseract 5.5.3\n")

    def test_official_windows_package_version_fails_closed_on_other_build(self):
        def fake_runner(args, **kwargs):
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=b"tesseract v5.5.3.99999999\n",
                stderr=b"",
            )

        runner = evidence._OfficialWindowsPackageRunner("tesseract.exe", runner=fake_runner)
        with self.assertRaises(RuntimeError):
            runner(["tesseract.exe", "--version"], capture_output=True, check=False)

    def test_package_runner_leaves_ocr_process_untouched(self):
        original = subprocess.CompletedProcess(
            ["tesseract.exe", "stdin", "stdout"],
            0,
            stdout=b"tsv",
            stderr=b"",
        )

        def fake_runner(args, **kwargs):
            return original

        runner = evidence._OfficialWindowsPackageRunner("tesseract.exe", runner=fake_runner)
        self.assertIs(runner(["tesseract.exe", "stdin", "stdout"]), original)
        self.assertIsNone(runner.reported_version)

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_work_dir_requires_runner_temp(self):
        with self.assertRaises(RuntimeError):
            evidence._bounded_work_dir(evidence.Path("child"))


if __name__ == "__main__":
    unittest.main()
