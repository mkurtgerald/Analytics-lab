import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from analytics_lab import face_donor_facessd_litert_probe as probe


def wheel_bytes(metadata: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("demo-1.0.dist-info/METADATA", metadata)
    return buffer.getvalue()


class LiteRTDependencyProbeTests(unittest.TestCase):
    def test_exact_converter_identity_is_pinned(self):
        self.assertEqual(probe._CONVERTER_VERSION, "0.4.0")
        self.assertEqual(
            probe._CONVERTER_SOURCE_COMMIT,
            "7d683c3c1104c29a4777d7047cff2fbe92bacce3",
        )
        self.assertEqual(
            probe._CONVERTER_WHEEL_SHA256,
            "77827e51886bca2fea2e56ce98e1fcc66f7985260c9b6bb6786ede1374638b51",
        )

    def test_inspect_wheel_reads_identity_license_and_requirements(self):
        payload = (
            b"Metadata-Version: 2.4\n"
            b"Name: demo\nVersion: 1.2.3\n"
            b"License: Apache-2.0\n"
            b"Classifier: License :: OSI Approved :: Apache Software License\n"
            b"Requires-Dist: numpy>=1.23\n\n"
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "demo-1.2.3-py3-none-any.whl"
            path.write_bytes(wheel_bytes(payload))
            result = probe._inspect_wheel(path)
        self.assertEqual(result["name"], "demo")
        self.assertEqual(result["version"], "1.2.3")
        self.assertEqual(result["license_fields"], ["Apache-2.0"])
        self.assertEqual(result["requires_dist"], ["numpy>=1.23"])

    def test_inspect_wheel_rejects_missing_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "demo-1.0-py3-none-any.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("demo.txt", b"x")
            with self.assertRaisesRegex(RuntimeError, "METADATA"):
                probe._inspect_wheel(path)

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_work_dir_requires_runner_temp(self):
        with self.assertRaisesRegex(RuntimeError, "RUNNER_TEMP"):
            probe._bounded_work_dir("child")


if __name__ == "__main__":
    unittest.main()
