from __future__ import annotations

from pathlib import Path
import unittest

from analytics_lab.figshare_acquisition import ZipMember
from analytics_lab.figshare_generalization3_admission import (
    ADL_NEGATIVE,
    FALL_POSITIVE,
    evidence_document,
    select_pinned_pair,
)
from analytics_lab.figshare_member_admission import AdmittedMember


class FigshareGeneralization3Tests(unittest.TestCase):
    def test_selects_only_exact_measured_pair(self) -> None:
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
        self.assertEqual([member.name for _, member in selected], [ADL_NEGATIVE.name, FALL_POSITIVE.name])

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

    def test_evidence_document_preserves_provenance_and_no_accuracy_claim(self) -> None:
        admitted = (
            AdmittedMember(
                role="negative",
                name=ADL_NEGATIVE.name,
                compression_method=8,
                compressed_size=ADL_NEGATIVE.compressed_size,
                uncompressed_size=ADL_NEGATIVE.uncompressed_size,
                crc32=ADL_NEGATIVE.crc32,
                sha256="1" * 64,
                local_path=Path("negative.mp4"),
            ),
            AdmittedMember(
                role="positive",
                name=FALL_POSITIVE.name,
                compression_method=8,
                compressed_size=FALL_POSITIVE.compressed_size,
                uncompressed_size=FALL_POSITIVE.uncompressed_size,
                crc32=FALL_POSITIVE.crc32,
                sha256="2" * 64,
                local_path=Path("positive.mp4"),
            ),
        )
        payload = evidence_document(admitted)
        self.assertEqual(payload["pair_id"], "generalization-3")
        self.assertEqual(payload["source"]["license_id"], "CC-BY-4.0")
        self.assertTrue(payload["source"]["attribution_required"])
        self.assertFalse(payload["media_retained_in_repository"])
        self.assertFalse(payload["commercial_accuracy_claim"])
        self.assertEqual([item["sha256"] for item in payload["members"]], ["1" * 64, "2" * 64])


if __name__ == "__main__":
    unittest.main()
