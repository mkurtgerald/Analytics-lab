from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from analytics_lab.figshare_acquisition import ZipMember
from analytics_lab.figshare_generalization4_admission import ADL_NEGATIVE, FALL_POSITIVE, evidence_document, select_pinned_pair
from analytics_lab.figshare_generalization4_person_down_diagnostic import _EXPECTED_SHA256, _sample_spec
from analytics_lab.figshare_member_admission import AdmittedMember
from analytics_lab.pose_adaptation_readiness import LIGHTWEIGHT_OPENPOSE_PLAN


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
            AdmittedMember("negative", ADL_NEGATIVE.name, 8, ADL_NEGATIVE.compressed_size, ADL_NEGATIVE.uncompressed_size, ADL_NEGATIVE.crc32, _EXPECTED_SHA256["negative"], Path("negative.mp4")),
            AdmittedMember("positive", FALL_POSITIVE.name, 8, FALL_POSITIVE.compressed_size, FALL_POSITIVE.uncompressed_size, FALL_POSITIVE.crc32, _EXPECTED_SHA256["positive"], Path("positive.mp4")),
        )
        payload = evidence_document(admitted)
        self.assertEqual(payload["pair_id"], "generalization-4")
        self.assertEqual(payload["source"]["license_id"], "CC-BY-4.0")
        self.assertTrue(payload["source"]["attribution_required"])
        self.assertFalse(payload["media_retained_in_repository"])
        self.assertFalse(payload["commercial_accuracy_claim"])
        self.assertEqual([item["sha256"] for item in payload["members"]], [_EXPECTED_SHA256["negative"], _EXPECTED_SHA256["positive"]])

    def test_exact_sha256_values_are_pinned(self) -> None:
        self.assertEqual(_EXPECTED_SHA256["negative"], "5fe01e92aa7705a9095b5b3b6906aeeea87d34d6f72c5493c2b606f53f14ea06")
        self.assertEqual(_EXPECTED_SHA256["positive"], "a88bed7d467a0129add7e1bf2ee5dc7de4a307bd646e8a100add52798660dd63")

    def test_sample_specs_bind_pinned_sha_and_distinct_sites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            negative = AdmittedMember("negative", ADL_NEGATIVE.name, 8, ADL_NEGATIVE.compressed_size, ADL_NEGATIVE.uncompressed_size, ADL_NEGATIVE.crc32, _EXPECTED_SHA256["negative"], root / "negative.mp4")
            positive = AdmittedMember("positive", FALL_POSITIVE.name, 8, FALL_POSITIVE.compressed_size, FALL_POSITIVE.uncompressed_size, FALL_POSITIVE.crc32, _EXPECTED_SHA256["positive"], root / "positive.mp4")
            negative_spec = _sample_spec(negative)
            positive_spec = _sample_spec(positive)
        self.assertEqual(negative_spec.media_sha256, _EXPECTED_SHA256["negative"])
        self.assertEqual(positive_spec.media_sha256, _EXPECTED_SHA256["positive"])
        self.assertEqual(negative_spec.site_id, "figshare-28596332-location-1")
        self.assertEqual(positive_spec.site_id, "figshare-28596332-location-3")
        self.assertEqual(negative_spec.labels, ())
        self.assertEqual(positive_spec.labels, ())

    def test_changed_valid_sha_fails_closed(self) -> None:
        item = AdmittedMember("negative", ADL_NEGATIVE.name, 8, ADL_NEGATIVE.compressed_size, ADL_NEGATIVE.uncompressed_size, ADL_NEGATIVE.crc32, "f" * 64, Path("negative.mp4"))
        self.assertNotEqual(item.sha256, _EXPECTED_SHA256["negative"])
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            _sample_spec(item)

    def test_malformed_sha_fails_before_measurement(self) -> None:
        item = AdmittedMember("negative", ADL_NEGATIVE.name, 8, ADL_NEGATIVE.compressed_size, ADL_NEGATIVE.uncompressed_size, ADL_NEGATIVE.crc32, "not-a-digest", Path("negative.mp4"))
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            _sample_spec(item)

    def test_selected_subjects_are_locked_out_of_pose_adaptation(self) -> None:
        holdouts = set(LIGHTWEIGHT_OPENPOSE_PLAN.holdout_subjects)
        self.assertIn("Figshare:SBJ_29", holdouts)
        self.assertIn("Figshare:SBJ_07", holdouts)
        self.assertFalse({"Figshare:SBJ_29", "Figshare:SBJ_07"} & set(LIGHTWEIGHT_OPENPOSE_PLAN.train_subjects))
        self.assertFalse({"Figshare:SBJ_29", "Figshare:SBJ_07"} & set(LIGHTWEIGHT_OPENPOSE_PLAN.validation_subjects))


if __name__ == "__main__":
    unittest.main()
