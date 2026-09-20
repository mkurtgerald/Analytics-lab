import hashlib
import re
import unittest
from unittest.mock import patch

from analytics_lab import lpr_front_envelope_evidence as evidence


class LPRFrontEnvelopeEvidenceTests(unittest.TestCase):
    def test_source_identity_discovers_then_enforces_hashes(self) -> None:
        payload = b"front-envelope-evidence"
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

    def test_unpinned_sha256_is_admission_only(self) -> None:
        identity = {
            "bytes": evidence._SOURCE_SIZE,
            "sha1": evidence._SOURCE_SHA1,
            "sha256": "b" * 64,
        }
        with patch.object(
            evidence, "_SOURCE_SHA256", None
        ), patch.object(
            evidence, "_fetch_bytes", return_value=b"placeholder"
        ), patch.object(evidence, "validate_source_bytes", return_value=identity):
            result = evidence.run()
        self.assertTrue(result["admission_only"])
        self.assertFalse(result["inference_run"])
        self.assertEqual(result["expected_text_normalized"], "MPR318")
        self.assertTrue(result["within_documented_plate_size_envelope"])
        self.assertEqual(
            result["evidence_scope"],
            "single_cc0_front_envelope_smoke_not_commercial_accuracy",
        )

    def test_rights_bounds_and_premeasurement_envelope_are_pinned(self) -> None:
        self.assertIn("oldid=1230217363", evidence._RIGHTS_PAGE)
        self.assertEqual(evidence._SOURCE_SIZE, 2_680_061)
        self.assertEqual((evidence._WIDTH, evidence._HEIGHT), (4_032, 3_024))
        self.assertEqual(evidence._MAX_SOURCE_BYTES, 3 * 1024 * 1024)
        self.assertIn("upload.wikimedia.org", evidence._SOURCE_URL)
        self.assertEqual(
            evidence._SOURCE_SHA1,
            "8cdb3acc024e67267dabf6d3ac793d0c54924e85",
        )
        if evidence._SOURCE_SHA256 is not None:
            self.assertRegex(evidence._SOURCE_SHA256, re.compile(r"[0-9a-f]{64}\Z"))
        self.assertEqual(evidence._DOCUMENTED_MIN_PLATE_WIDTH_PX, 96)
        self.assertGreaterEqual(
            evidence._PREDECLARED_PLATE_WIDTH_LOWER_BOUND_PX,
            evidence._DOCUMENTED_MIN_PLATE_WIDTH_PX,
        )


if __name__ == "__main__":
    unittest.main()
