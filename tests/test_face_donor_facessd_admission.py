import io
import tarfile
import unittest
from unittest import mock

from analytics_lab import face_donor_facessd_admission as admission


class FaceSSDAdmissionTests(unittest.TestCase):
    def test_safe_members_rejects_traversal(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("../escape.tflite")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            admission._safe_members(buffer.getvalue())

    def test_safe_members_accepts_bounded_model_files(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, payload in (
                ("model/detect.tflite", b"abc"),
                ("model/pipeline.config", b"cfg"),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        members = admission._safe_members(buffer.getvalue())
        self.assertEqual([m["name"] for m in members], [
            "model/detect.tflite", "model/pipeline.config"
        ])

    @mock.patch("analytics_lab.face_donor_facessd_admission._download", return_value=b"not-a-tar")
    def test_run_fails_closed_on_invalid_archive(self, _download):
        with self.assertRaises((tarfile.ReadError, RuntimeError)):
            admission.run()


if __name__ == "__main__":
    unittest.main()
