from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from analytics_lab.figshare_acquisition import ZipMember
from analytics_lab.figshare_generalization4_admission import ADL_NEGATIVE, FALL_POSITIVE, evidence_document, select_pinned_pair
from analytics_lab.figshare_generalization4_person_down_diagnostic import _EXPECTED_SHA256, _sample_spec
from analytics_lab.figshare_member_admission import AdmittedMember


class FigshareGeneralization4Tests(unittest.TestCase):
    def test_selects_only_exact_measured_pair(self) -> None:
        members = (
            ZipMember(ADL_NEGATIVE.name, ADL_NEGATIVE.compressed_size, ADL_NEGATIVE.uncompressed_size, 8, ADL_NEGATIVE.crc32, ADL_NEGATIVE.local_header_offset),
            ZipMember(FALL_POSITIVE.name, FALL_POSITIVE.compressed_size, FALL_POSITIVE.uncompressed_size, 8, FALL_POSITIVE.crc32, FALL_POSITIVE.local_header_offset),
        )
        selected = select_pinned_pair(members)
        self.assertEqual([item.role for item, _ in selected], ["negative", "positive"])
        self.assertEqual([member.name for _, member in selected], [ADL_NEGATIVE.name, FALL_POSITIVE.name])

    def test_changed_archive_identity_fails_closed(self) -> None:
        members = (
            ZipMember(ADL_NEGATIVE.name, ADL_NEGATIVE.compressed_size + 1, ADL_NEGATIVE.uncompressed_size, 8, ADL_NEGATIVE.crc32, ADL_NEGATIVE.local_header_offset),
            ZipMember(FALL_POSITIVE.name, FALL_POSITIVE.compressed_size, FALL_POSITIVE.uncompressed_size, 8, FALL_POSITIVE.crc32, FALL_POSITIVE.local_header_offset),
        )
        with self.assertRaisesRegex(RuntimeError, "missing or changed"):
            select_pinned_pair(members)

    def test_evidence_document_preserves_provenance_and_no_accuracy_claim(self) -> None:
        admitted = (
            AdmittedMember("negative", ADL_NEGATIVE.name, 8, ADL_NEGATIVE.compressed_size, ADL_NEGATIVE.uncompressed_size, ADL_NEGATIVE.crc32, "1" * 64, Path("negative.mp4")),
            AdmittedMember("positive", FALL_POSITIVE.name, 8, FALL_POSITIVE.compressed_size, FALL_POSITIVE.uncompressed_size, FALL_POSITIVE.crc32, "2" * 64, Path("positive.mp4")),
        )
        payload = evidence_document(admitted)
        self.assertEqual(payload["pair_id"], "generalization-4")
        self.assertEqual(payload["source"]["license_id"], "CC-BY-4.0")
        self.assertTrue(payload["source"]["attribution_required"])
        self.assertFalse(payload["media_retained_in_repository"])
        self.assertFalse(payload["commercial_accuracy_claim"])

    def test_sample_specs_bind_runtime_digest_and_distinct_sites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            negative_sha = _EXPECTED_SHA256["negative"] or ("1" * 64)
            positive_sha = _EXPECTED_SHA256["positive"] or ("2" * 64)
            negative = AdmittedMember("negative", ADL_NEGATIVE.name, 8, ADL_NEGATIVE.compressed_size, ADL_NEGATIVE.uncompressed_size, ADL_NEGATIVE.crc32, negative_sha, root / "negative.mp4")
            positive = AdmittedMember("positive", FALL_POSITIVE.name, 8, FALL_POSITIVE.compressed_size, FALL_POSITIVE.uncompressed_size, FALL_POSITIVE.crc32, positive_sha, root / "positive.mp4")
            negative_spec = _sample_spec(negative)
            positive_spec = _sample_spec(positive)
        self.assertEqual(negative_spec.site_id, "figshare-28596332-location-1")
        self.assertEqual(positive_spec.site_id, "figshare-28596332-location-3")
        self.assertEqual(negative_spec.labels, ())
        self.assertEqual(positive_spec.labels, ())

    def test_malformed_sha_fails_before_measurement(self) -> None:
        item = AdmittedMember("negative", ADL_NEGATIVE.name, 8, ADL_NEGATIVE.compressed_size, ADL_NEGATIVE.uncompressed_size, ADL_NEGATIVE.crc32, "not-a-digest", Path("negative.mp4"))
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            _sample_spec(item)


if __name__ == "__main__":
    unittest.main()
