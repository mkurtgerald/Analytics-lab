import hashlib
import io
import tarfile
import unittest
from unittest import mock

from analytics_lab import face_donor_oid_ssd_admission as admission


class OIDSSDAdmissionTests(unittest.TestCase):
    def test_archive_and_complete_member_identity_are_pinned(self):
        self.assertEqual(admission._EXPECTED_ARCHIVE_SIZE, 158_851_107)
        self.assertEqual(
            admission._EXPECTED_ARCHIVE_SHA256,
            "8dd82cc52625eb9b43c6ed8050868fe1e78f878c6f028519fc731b3c7e9035f7",
        )
        self.assertEqual(
            admission._EXPECTED_ARCHIVE_MEMBERS,
            (
                (
                    "ssd_mobilenet_v2_oid_v4_2018_12_12/model.ckpt.meta",
                    11_915_505,
                    "541aebeef5c674609f4b2cc229c265b053dc7df3226143b01037c0a42d715f7f",
                ),
                (
                    "ssd_mobilenet_v2_oid_v4_2018_12_12/checkpoint",
                    77,
                    "dd1b025d2e155283f5e300ce95bf6d5b6bc0f7fe010db73daa6975eb896ab9cb",
                ),
                (
                    "ssd_mobilenet_v2_oid_v4_2018_12_12/frozen_inference_graph.pb",
                    66_606_111,
                    "150f3eb77b741ed24e1a19559589205ccd616b059fc976c8b8d5cefc126bf86b",
                ),
                (
                    "ssd_mobilenet_v2_oid_v4_2018_12_12/saved_model/saved_model.pb",
                    67_889_548,
                    "f5453b9c2bb73be4d21eef9eb37a8fce0a2d09903ea1aecd44ff49fa6efe258f",
                ),
                (
                    "ssd_mobilenet_v2_oid_v4_2018_12_12/model.ckpt.index",
                    14_175,
                    "0557da0b4b7d555fc1483d3ba3a89732c7606e7176c30839335333f48b9d81f4",
                ),
                (
                    "ssd_mobilenet_v2_oid_v4_2018_12_12/pipeline.config",
                    4_267,
                    "cf424b06dabcc7acd6bf71ffd941c5a9890b975f8036444e03a2290391a24614",
                ),
                (
                    "ssd_mobilenet_v2_oid_v4_2018_12_12/model.ckpt.data-00000-of-00001",
                    57_841_536,
                    "61bc5931d1cc44cd83b80ebee3cb75757e84c1a49ac5cea71149e36a93da4044",
                ),
            ),
        )

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
        self.assertIsNone(admission._ARCHIVE_LABEL_MAP_MEMBER)
        self.assertEqual(admission._HUMAN_FACE_CLASS_ID, 502)
        self.assertEqual(admission._HUMAN_FACE_MID, "/m/0dzct")
        self.assertEqual(admission._HUMAN_FACE_DISPLAY_NAME, "Human face")
        self.assertIn("verify each image license", admission._OPEN_IMAGES_LICENSE_CAVEAT)

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

    def test_run_can_still_represent_unpinned_discovery_under_explicit_patch(self):
        payload = b"synthetic tar bytes"
        members = [
            {
                "name": "model/frozen_inference_graph.pb",
                "size": 5,
                "type": "file",
                "sha256": hashlib.sha256(b"graph").hexdigest(),
            },
            {
                "name": "model/saved_model.pb",
                "size": 5,
                "type": "file",
                "sha256": hashlib.sha256(b"saved").hexdigest(),
            },
            {
                "name": "model/pipeline.config",
                "size": 3,
                "type": "file",
                "sha256": hashlib.sha256(b"cfg").hexdigest(),
            },
        ]
        identity = admission._member_identity(members)
        sha256 = hashlib.sha256(payload).hexdigest()
        with (
            mock.patch.object(admission, "_declared_size", return_value=len(payload)),
            mock.patch.object(admission, "_download", return_value=payload),
            mock.patch.object(admission, "_safe_members", return_value=members),
            mock.patch.object(admission, "_EXPECTED_ARCHIVE_SIZE", len(payload)),
            mock.patch.object(admission, "_EXPECTED_ARCHIVE_SHA256", sha256),
            mock.patch.object(admission, "_EXPECTED_ARCHIVE_MEMBERS", identity),
        ):
            result = admission.run()
        self.assertTrue(result["admitted"])
        self.assertTrue(result["archive_pinned"])
        self.assertTrue(result["members_pinned"])
        self.assertFalse(result["inference_run"])
        self.assertFalse(result["media_used"])
        self.assertFalse(result["artifact_retained"])
        self.assertIsNone(result["provenance"]["archive_label_map_member"])

    def test_run_fails_closed_on_hash_mismatch(self):
        payload = b"synthetic tar bytes"
        members = [
            {
                "name": "model/frozen_inference_graph.pb",
                "size": 5,
                "type": "file",
                "sha256": hashlib.sha256(b"graph").hexdigest(),
            },
            {
                "name": "model/saved_model.pb",
                "size": 5,
                "type": "file",
                "sha256": hashlib.sha256(b"saved").hexdigest(),
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

    def test_run_fails_closed_if_archive_suddenly_contains_label_map(self):
        payload = b"synthetic tar bytes"
        members = [
            {
                "name": "model/frozen_inference_graph.pb",
                "size": 5,
                "type": "file",
                "sha256": hashlib.sha256(b"graph").hexdigest(),
            },
            {
                "name": "model/saved_model.pb",
                "size": 5,
                "type": "file",
                "sha256": hashlib.sha256(b"saved").hexdigest(),
            },
            {
                "name": "model/pipeline.config",
                "size": 3,
                "type": "file",
                "sha256": hashlib.sha256(b"cfg").hexdigest(),
            },
            {
                "name": "model/label_map.pbtxt",
                "size": 4,
                "type": "file",
                "sha256": hashlib.sha256(b"map!").hexdigest(),
            },
        ]
        identity = admission._member_identity(members)
        sha256 = hashlib.sha256(payload).hexdigest()
        with (
            mock.patch.object(admission, "_declared_size", return_value=len(payload)),
            mock.patch.object(admission, "_download", return_value=payload),
            mock.patch.object(admission, "_safe_members", return_value=members),
            mock.patch.object(admission, "_EXPECTED_ARCHIVE_SIZE", len(payload)),
            mock.patch.object(admission, "_EXPECTED_ARCHIVE_SHA256", sha256),
            mock.patch.object(admission, "_EXPECTED_ARCHIVE_MEMBERS", identity),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected.*label-map"):
                admission.run()


if __name__ == "__main__":
    unittest.main()
