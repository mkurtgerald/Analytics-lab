import hashlib
import io
import tarfile
import unittest
from unittest import mock

from analytics_lab import face_donor_facessd_admission as admission


class FaceSSDAdmissionTests(unittest.TestCase):
    def test_archive_identity_is_pinned(self):
        self.assertEqual(admission._EXPECTED_ARCHIVE_SIZE, 130_655_026)
        self.assertEqual(
            admission._EXPECTED_ARCHIVE_SHA256,
            "9ae49a245caddbe7d7bbc82a35da0191a2f2e210161df19be357a1c7f49118d5",
        )

    def test_safe_members_rejects_traversal(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("../escape.tflite")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            admission._safe_members(buffer.getvalue())

    def test_safe_members_hashes_bounded_model_files(self):
        buffer = io.BytesIO()
        expected = {}
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, payload in (
                ("model/detect.tflite", b"abc"),
                ("model/pipeline.config", b"cfg"),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
                expected[name] = hashlib.sha256(payload).hexdigest()
        members = admission._safe_members(buffer.getvalue())
        self.assertEqual([m["name"] for m in members], [
            "model/detect.tflite", "model/pipeline.config"
        ])
        self.assertEqual(
            [m["sha256"] for m in members],
            [expected["model/detect.tflite"], expected["model/pipeline.config"]],
        )

    @mock.patch(
        "analytics_lab.face_donor_facessd_admission._declared_size",
        return_value=admission._EXPECTED_ARCHIVE_SIZE + 1,
    )
    def test_run_fails_closed_on_declared_size_mismatch(self, _declared):
        with mock.patch(
            "analytics_lab.face_donor_facessd_admission._download"
        ) as download:
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                admission.run()
        download.assert_not_called()

    @mock.patch(
        "analytics_lab.face_donor_facessd_admission._declared_size",
        return_value=admission._EXPECTED_ARCHIVE_SIZE,
    )
    @mock.patch(
        "analytics_lab.face_donor_facessd_admission._download",
        return_value=b"wrong archive",
    )
    def test_run_fails_closed_on_archive_hash_mismatch(self, _download, _declared):
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            admission.run()


if __name__ == "__main__":
    unittest.main()
