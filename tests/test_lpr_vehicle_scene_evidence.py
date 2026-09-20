import hashlib
import unittest
from unittest.mock import patch

from analytics_lab import lpr_vehicle_scene_evidence as evidence


class LPRVehicleSceneEvidenceTests(unittest.TestCase):
    def test_source_identity_discovers_then_enforces_hashes(self) -> None:
        payload = b"vehicle-scene-evidence"
        sha1 = hashlib.sha1(payload).hexdigest()
        sha256 = hashlib.sha256(payload).hexdigest()
        with patch.object(evidence, "_SOURCE_SIZE", len(payload)):
            identity = evidence.validate_source_bytes(
                payload, required_sha1=None, required_sha256=None
            )
            self.assertEqual(identity["sha1"], sha1)
            self.assertEqual(identity["sha256"], sha256)
            self.assertEqual(
                evidence.validate_source_bytes(
                    payload, required_sha1=sha1, required_sha256=sha256
                )["bytes"],
                len(payload),
            )
            with self.assertRaisesRegex(RuntimeError, "SHA-1"):
                evidence.validate_source_bytes(
                    payload, required_sha1="0" * 40, required_sha256=sha256
                )
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                evidence.validate_source_bytes(
                    payload, required_sha1=sha1, required_sha256="0" * 64
                )
        with self.assertRaisesRegex(RuntimeError, "size"):
            evidence.validate_source_bytes(
                payload, required_sha1=None, required_sha256=None
            )

    def test_first_head_is_admission_only(self) -> None:
        identity = {
            "bytes": evidence._SOURCE_SIZE,
            "sha1": "a" * 40,
            "sha256": "b" * 64,
        }
        with patch.object(evidence, "_SOURCE_SHA1", None), patch.object(
            evidence, "_SOURCE_SHA256", None
        ), patch.object(
            evidence, "_fetch_bytes", return_value=b"placeholder"
        ), patch.object(evidence, "validate_source_bytes", return_value=identity):
            result = evidence.run()
        self.assertTrue(result["admission_only"])
        self.assertFalse(result["inference_run"])
        self.assertEqual(result["expected_text_normalized"], "BK0074")
        self.assertEqual(
            result["evidence_scope"],
            "single_public_domain_vehicle_scene_smoke_not_commercial_accuracy",
        )

    def test_rights_source_bounds_and_source_hashes_are_pinned(self) -> None:
        self.assertIn("oldid=1109767728", evidence._RIGHTS_PAGE)
        self.assertEqual(evidence._SOURCE_SIZE, 26_397)
        self.assertEqual((evidence._WIDTH, evidence._HEIGHT), (538, 349))
        self.assertEqual(evidence._MAX_SOURCE_BYTES, 32_768)
        self.assertIn("upload.wikimedia.org", evidence._SOURCE_URL)
        self.assertEqual(
            evidence._SOURCE_SHA1,
            "84ac1c7c66b10345fff12e7ed6876c46b91abbd3",
        )
        self.assertEqual(
            evidence._SOURCE_SHA256,
            "f85e058f4526c43b34f97edf4349b8510b321db3fb98e5ce7a47b7f579e6f890",
        )


if __name__ == "__main__":
    unittest.main()
