from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from analytics_lab.figshare_acquisition import ZipMember
from analytics_lab.figshare_generalization2_admission import (
    ADL_NEGATIVE,
    FALL_POSITIVE,
    select_pinned_pair,
)
from analytics_lab.figshare_generalization2_person_down_diagnostic import _sample_spec
from analytics_lab.figshare_member_admission import AdmittedMember


class FigshareGeneralization2Tests(unittest.TestCase):
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

    def test_sample_specs_bind_computed_sha_and_distinct_sites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            negative = AdmittedMember(
                role="negative",
                name=ADL_NEGATIVE.name,
                compression_method=8,
                compressed_size=ADL_NEGATIVE.compressed_size,
                uncompressed_size=ADL_NEGATIVE.uncompressed_size,
                crc32=ADL_NEGATIVE.crc32,
                sha256="a" * 64,
                local_path=root / "negative.mp4",
            )
            positive = AdmittedMember(
                role="positive",
                name=FALL_POSITIVE.name,
                compression_method=8,
                compressed_size=FALL_POSITIVE.compressed_size,
                uncompressed_size=FALL_POSITIVE.uncompressed_size,
                crc32=FALL_POSITIVE.crc32,
                sha256="b" * 64,
                local_path=root / "positive.mp4",
            )
            negative_spec = _sample_spec(negative)
            positive_spec = _sample_spec(positive)

        self.assertEqual(negative_spec.media_sha256, "a" * 64)
        self.assertEqual(positive_spec.media_sha256, "b" * 64)
        self.assertEqual(negative_spec.site_id, "figshare-28596332-location-1")
        self.assertEqual(positive_spec.site_id, "figshare-28596332-location-2")
        self.assertEqual(negative_spec.labels, ())
        self.assertEqual(positive_spec.labels, ())

    def test_malformed_sha_fails_before_measurement(self) -> None:
        item = AdmittedMember(
            role="negative",
            name=ADL_NEGATIVE.name,
            compression_method=8,
            compressed_size=ADL_NEGATIVE.compressed_size,
            uncompressed_size=ADL_NEGATIVE.uncompressed_size,
            crc32=ADL_NEGATIVE.crc32,
            sha256="not-a-digest",
            local_path=Path("negative.mp4"),
        )
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            _sample_spec(item)


if __name__ == "__main__":
    unittest.main()
