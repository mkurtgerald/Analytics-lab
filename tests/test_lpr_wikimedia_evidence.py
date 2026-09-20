import hashlib
import unittest
from unittest.mock import patch

from analytics_lab import lpr_wikimedia_evidence as evidence
from analytics_lab.lpr_ocr import PlateDetection


class LPRWikimediaEvidenceTests(unittest.TestCase):
    def test_source_identity_rejects_size_or_sha_mismatch(self) -> None:
        payload = b"bounded-evidence"
        with patch.object(evidence, "_SOURCE_SIZE", len(payload)), patch.object(
            evidence, "_SOURCE_SHA1", hashlib.sha1(payload).hexdigest()
        ):
            identity = evidence.validate_source_bytes(payload, required_sha256=None)
            self.assertEqual(identity["bytes"], len(payload))
            self.assertEqual(identity["sha256"], hashlib.sha256(payload).hexdigest())
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                evidence.validate_source_bytes(payload, required_sha256="0" * 64)
        with self.assertRaisesRegex(RuntimeError, "size"):
            evidence.validate_source_bytes(payload, required_sha256=None)

    def test_full_frame_iou_is_bounded(self) -> None:
        self.assertAlmostEqual(
            evidence._iou_full_frame(PlateDetection(0.9, 0.1, 0.2, 0.9, 0.8)),
            0.48,
        )
        self.assertEqual(
            evidence._iou_full_frame(PlateDetection(0.9, 0.0, 0.0, 1.0, 1.0)),
            1.0,
        )

    def test_first_head_is_admission_only(self) -> None:
        identity = {
            "bytes": evidence._SOURCE_SIZE,
            "sha1": evidence._SOURCE_SHA1,
            "sha256": "a" * 64,
        }
        with patch.object(evidence, "_SOURCE_SHA256", None), patch.object(
            evidence, "_fetch_bytes", return_value=b"placeholder"
        ), patch.object(evidence, "validate_source_bytes", return_value=identity):
            result = evidence.run()
        self.assertTrue(result["admission_only"])
        self.assertFalse(result["inference_run"])
        self.assertEqual(result["expected_text_normalized"], "A92518")
        self.assertEqual(
            result["evidence_scope"],
            "staged_public_domain_smoke_not_commercial_accuracy",
        )

    def test_rights_and_source_are_pinned(self) -> None:
        self.assertIn("oldid=856647637", evidence._RIGHTS_PAGE)
        self.assertEqual(evidence._SOURCE_SIZE, 61_465)
        self.assertEqual(evidence._SOURCE_SHA1, "d365a117631a8fa5a2a0fb7a8d2a03fe2e9b73bc")
        self.assertEqual((evidence._WIDTH, evidence._HEIGHT), (680, 144))


if __name__ == "__main__":
    unittest.main()
