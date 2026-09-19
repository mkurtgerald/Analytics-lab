from __future__ import annotations

from pathlib import Path
import unittest

from analytics_lab.figshare_acquisition import ZipMember
from analytics_lab.figshare_generalization5_admission import (
    ADL_NEGATIVE,
    EXPECTED_SHA256,
    FALL_POSITIVE,
    evidence_document,
    select_pinned_pair,
)
from analytics_lab.figshare_generalization5_person_down_diagnostic import _sample_spec
from analytics_lab.figshare_member_admission import AdmittedMember
from analytics_lab.pose_adaptation_readiness import LIGHTWEIGHT_OPENPOSE_PLAN


class FigshareGeneralization5Tests(unittest.TestCase):
    def _admitted(self, role: str, sha256: str) -> AdmittedMember:
        member = ADL_NEGATIVE if role == "negative" else FALL_POSITIVE
        return AdmittedMember(
            role,
            member.name,
            8,
            member.compressed_size,
            member.uncompressed_size,
            member.crc32,
            sha256,
            Path(role + ".mp4"),
        )

    def test_selects_only_exact_predeclared_pair(self) -> None:
        members = (
            ZipMember(
                ADL_NEGATIVE.name,
                ADL_NEGATIVE.compressed_size,
                ADL_NEGATIVE.uncompressed_size,
                8,
                ADL_NEGATIVE.crc32,
                ADL_NEGATIVE.local_header_offset,
            ),
            ZipMember(
                FALL_POSITIVE.name,
                FALL_POSITIVE.compressed_size,
                FALL_POSITIVE.uncompressed_size,
                8,
                FALL_POSITIVE.crc32,
                FALL_POSITIVE.local_header_offset,
            ),
        )
        selected = select_pinned_pair(members)
        self.assertEqual([item.role for item, _ in selected], ["negative", "positive"])
        self.assertEqual(
            [member.name for _, member in selected],
            [ADL_NEGATIVE.name, FALL_POSITIVE.name],
        )

    def test_changed_archive_identity_fails_closed(self) -> None:
        members = (
            ZipMember(
                ADL_NEGATIVE.name,
                ADL_NEGATIVE.compressed_size + 1,
                ADL_NEGATIVE.uncompressed_size,
                8,
                ADL_NEGATIVE.crc32,
                ADL_NEGATIVE.local_header_offset,
            ),
            ZipMember(
                FALL_POSITIVE.name,
                FALL_POSITIVE.compressed_size,
                FALL_POSITIVE.uncompressed_size,
                8,
                FALL_POSITIVE.crc32,
                FALL_POSITIVE.local_header_offset,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "missing or changed"):
            select_pinned_pair(members)

    def test_exact_sha256_values_are_pinned(self) -> None:
        self.assertEqual(
            EXPECTED_SHA256,
            {
                "negative": "350587303666161ad00b812f69f64397b071e3b4d98b246556736943ca9c93f3",
                "positive": "8ec4976d74bdf4ac046eea4946d2fd65a4b65ad834c76d02d8488da7fa5406a1",
            },
        )

    def test_changed_valid_sha_fails_closed(self) -> None:
        changed = "0" * 64
        self.assertNotEqual(changed, EXPECTED_SHA256["negative"])
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            _sample_spec(self._admitted("negative", changed))

    def test_malformed_sha_fails_before_measurement(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            _sample_spec(self._admitted("positive", "bad"))

    def test_sample_specs_bind_pinned_sha_and_distinct_sites(self) -> None:
        negative = _sample_spec(self._admitted("negative", EXPECTED_SHA256["negative"]))
        positive = _sample_spec(self._admitted("positive", EXPECTED_SHA256["positive"]))
        self.assertEqual(negative.media_sha256, EXPECTED_SHA256["negative"])
        self.assertEqual(positive.media_sha256, EXPECTED_SHA256["positive"])
        self.assertNotEqual(negative.site_id, positive.site_id)
        self.assertFalse(negative.labels)
        self.assertFalse(positive.labels)

    def test_evidence_document_preserves_rights_and_no_accuracy_claim(self) -> None:
        admitted = (
            self._admitted("negative", EXPECTED_SHA256["negative"]),
            self._admitted("positive", EXPECTED_SHA256["positive"]),
        )
        payload = evidence_document(admitted)
        self.assertEqual(payload["pair_id"], "generalization-5")
        self.assertEqual(payload["source"]["license_id"], "CC-BY-4.0")
        self.assertTrue(payload["source"]["attribution_required"])
        self.assertTrue(payload["media_sha256_pre_pinned"])
        self.assertFalse(payload["media_retained_in_repository"])
        self.assertFalse(payload["commercial_accuracy_claim"])

    def test_selected_subjects_are_locked_out_of_pose_adaptation(self) -> None:
        holdouts = set(LIGHTWEIGHT_OPENPOSE_PLAN.holdout_subjects)
        self.assertIn("Figshare:SBJ_26", holdouts)
        self.assertIn("Figshare:SBJ_08", holdouts)
        self.assertFalse(
            {"Figshare:SBJ_26", "Figshare:SBJ_08"}
            & set(LIGHTWEIGHT_OPENPOSE_PLAN.train_subjects)
        )
        self.assertFalse(
            {"Figshare:SBJ_26", "Figshare:SBJ_08"}
            & set(LIGHTWEIGHT_OPENPOSE_PLAN.validation_subjects)
        )


if __name__ == "__main__":
    unittest.main()
