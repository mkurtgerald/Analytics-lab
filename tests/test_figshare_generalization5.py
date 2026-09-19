from __future__ import annotations

from pathlib import Path
import unittest

from analytics_lab.figshare_acquisition import ZipMember
from analytics_lab.figshare_generalization5_admission import (
    ADL_NEGATIVE,
    FALL_POSITIVE,
    evidence_document,
    select_pinned_pair,
)
from analytics_lab.figshare_member_admission import AdmittedMember
from analytics_lab.pose_adaptation_readiness import LIGHTWEIGHT_OPENPOSE_PLAN


class FigshareGeneralization5Tests(unittest.TestCase):
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

    def test_evidence_document_preserves_rights_and_no_accuracy_claim(self) -> None:
        admitted = (
            AdmittedMember(
                "negative",
                ADL_NEGATIVE.name,
                8,
                ADL_NEGATIVE.compressed_size,
                ADL_NEGATIVE.uncompressed_size,
                ADL_NEGATIVE.crc32,
                "1" * 64,
                Path("negative.mp4"),
            ),
            AdmittedMember(
                "positive",
                FALL_POSITIVE.name,
                8,
                FALL_POSITIVE.compressed_size,
                FALL_POSITIVE.uncompressed_size,
                FALL_POSITIVE.crc32,
                "2" * 64,
                Path("positive.mp4"),
            ),
        )
        payload = evidence_document(admitted)
        self.assertEqual(payload["pair_id"], "generalization-5")
        self.assertEqual(payload["source"]["license_id"], "CC-BY-4.0")
        self.assertTrue(payload["source"]["attribution_required"])
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
