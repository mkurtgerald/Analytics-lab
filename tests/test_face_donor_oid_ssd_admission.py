import hashlib
import io
import tarfile
import unittest
from unittest import mock

from analytics_lab import face_donor_oid_ssd_admission as admission


class OIDSSDAdmissionTests(unittest.TestCase):
    def test_provenance_and_human_face_mapping_are_pinned(self):
        self.assertEqual(
            admission._TF_MODELS_REVISION,
            "0558408514dacf2fe2860cd72ac56cbdf62a24c0",
        )
        self.assertEqual(admission._TF_MODELS_LICENSE, "Apache-2.0")
        self.assertEqual(
            admission._TF_MODELS_LICENSE_BLOB,
            "43fcf7bf1f1f9f824a1debf05d6ced45bf5810aa",
        )
        self.assertEqual(
            admission._OID_V4_LABEL_MAP_BLOB,
            "643b9e8ed5d9239a3248b895fb32f3b51caa92f3",
        )
        self.assertEqual(admission._HUMAN_FACE_CLASS_ID, 502)
        self.assertEqual(admission._HUMAN_FACE_MID, "/m/0dzct")
        self.assertEqual(admission._HUMAN_FACE_DISPLAY_NAME, "Human face")
        self.assertIn("verify each image license", admission._OPEN_IMAGES_LICENSE_CAVEAT)

    def test_discovery_head_is_not_admission(self):
        self.assertIsNone(admission._EXPECTED_ARCHIVE_SIZE)
        self.assertIsNone(admission._EXPECTED_ARCHIVE_SHA256)
        self.assertIsNone(admission._EXPECTED_ARCHIVE_MEMBERS)

    def test_safe_members_rejects_traversal(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("../escape.pb")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            admission._safe_members(buffer.getvalue())

    def test_safe_members_hashes_regular_files(self):
        buffer = io.BytesIO()
        expected = {}
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, payload in (
                ("model/frozen_inference_graph.pb", b"graph"),
                ("model/pipeline.config", b"cfg"),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
                expected[name] = hashlib.sha256(payload).hexdigest()
        members = admission._safe_members(buffer.getvalue())
        self.assertEqual(
            [m["sha256"] for m in members],
            [
                expected["model/frozen_inference_graph.pb"],
                expected["model/pipeline.config"],
            ],
        )

    def test_run_discovery_reports_not_admitted(self):
        payload = b"synthetic tar bytes"
        members = [
            {
                "name": "model/frozen_inference_graph.pb",
                "size": 5,
                "type": "file",
                "sha256": hashlib.sha256(b"graph").hexdigest(),
            },
            {
                "name": "model/pipeline.config",
                "size": 3,
                "type": "file",
                "sha256": hashlib.sha256(b"cfg").hexdigest(),
            },
        ]
        with (
            mock.patch.object(admission, "_declared_size", return_value=len(payload)),
            mock.patch.object(admission, "_download", return_value=payload),
            mock.patch.object(admission, "_safe_members", return_value=members),
        ):
            result = admission.run()
        self.assertFalse(result["admitted"])
        self.assertFalse(result["archive_pinned"])
        self.assertFalse(result["inference_run"])
        self.assertFalse(result["media_used"])
        self.assertFalse(result["artifact_retained"])

    def test_pinned_run_fails_closed_on_hash_mismatch(self):
        payload = b"synthetic tar bytes"
        members = [
            {
                "name": "model/frozen_inference_graph.pb",
                "size": 5,
                "type": "file",
                "sha256": hashlib.sha256(b"graph").hexdigest(),
            },
            {
                "name": "model/pipeline.config",
                "size": 3,
                "type": "file",
                "sha256": hashlib.sha256(b"cfg").hexdigest(),
            },
        ]
        identity = admission._member_identity(members)
        with (
            mock.patch.object(admission, "_declared_size", return_value=len(payload)),
            mock.patch.object(admission, "_download", return_value=payload),
            mock.patch.object(admission, "_safe_members", return_value=members),
            mock.patch.object(admission, "_EXPECTED_ARCHIVE_SIZE", len(payload)),
            mock.patch.object(admission, "_EXPECTED_ARCHIVE_SHA256", "0" * 64),
            mock.patch.object(admission, "_EXPECTED_ARCHIVE_MEMBERS", identity),
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                admission.run()


if __name__ == "__main__":
    unittest.main()
