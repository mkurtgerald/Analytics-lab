from __future__ import annotations

import struct
import unittest

from analytics_lab.figshare_acquisition import FigshareArtifact, ZipMember
from analytics_lab.figshare_probe import (
    DirectoryDescriptor,
    _github_escape,
    directory_ranges,
    inspect_directory_descriptor,
    summarize_descriptor,
    summarize_index,
)


class FigshareProbeTests(unittest.TestCase):
    def setUp(self):
        self.artifact = FigshareArtifact(
            file_id=52990358,
            name="VideoDataset.zip",
            size_bytes=4_000_000,
            download_url="https://figshare.com/ndownloader/files/52990358",
            md5="0123456789abcdef0123456789abcdef",
        )

    def test_summary_is_bounded_sorted_and_video_only(self):
        members = (
            ZipMember("subject-b/fall-large.mp4", 200, 300, 8, 0x1234, 40),
            ZipMember("subject-a/adl-small.mp4", 100, 120, 8, 0x5678, 10),
            ZipMember("subject-a/readme.txt", 10, 10, 8, 0x9ABC, 5),
            ZipMember("subject-c/too-large.mp4", 100, 17 * 1024 * 1024, 8, 0xDEF0, 80),
        )
        summary = summarize_index(self.artifact, members, sample_limit=2)
        self.assertEqual(summary["archive_member_count"], 4)
        self.assertEqual(summary["bounded_video_count"], 2)
        self.assertEqual([item["name"] for item in summary["smallest_bounded_videos"]], [
            "subject-a/adl-small.mp4",
            "subject-b/fall-large.mp4",
        ])
        self.assertEqual(summary["bounded_video_extensions"], {".mp4": 2})
        self.assertEqual(summary["smallest_bounded_videos"][0]["crc32"], "00005678")

    def test_summary_limit_is_hard_bounded(self):
        with self.assertRaises(ValueError):
            summarize_index(self.artifact, tuple(), sample_limit=33)

    def test_measured_shape_is_outside_old_ceiling_but_inside_probe_ceiling(self):
        descriptor = DirectoryDescriptor(22_397, 2_526_579_061, 2_941_785)
        self.assertFalse(descriptor.within_current_admission_ceiling)
        self.assertTrue(descriptor.within_probe_ceiling)
        ranges = directory_ranges(descriptor)
        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0][1] - ranges[0][0] + 1, 2 * 1024 * 1024)
        self.assertEqual(sum(end - start + 1 for start, end in ranges), descriptor.size)
        self.assertEqual(ranges[0][1] + 1, ranges[1][0])

    def test_probe_ceiling_remains_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "probe ceiling"):
            directory_ranges(DirectoryDescriptor(25_001, 100, 1))
        with self.assertRaisesRegex(ValueError, "probe ceiling"):
            directory_ranges(DirectoryDescriptor(10, 100, 4 * 1024 * 1024 + 1))

    def test_descriptor_observes_over_old_ceiling_without_admitting_member_payload(self):
        archive_size = 4_000_000
        central_size = 3 * 1024 * 1024
        central_offset = 500_000
        entry_count = 12_000
        eocd = struct.pack(
            "<4s4H2LH",
            b"PK\x05\x06",
            0,
            0,
            entry_count,
            entry_count,
            central_size,
            central_offset,
            0,
        )
        tail_start = archive_size - len(eocd)
        descriptor = inspect_directory_descriptor(
            eocd,
            tail_start=tail_start,
            archive_size=archive_size,
        )
        self.assertEqual(descriptor.entry_count, entry_count)
        self.assertEqual(descriptor.size, central_size)
        self.assertFalse(descriptor.within_current_admission_ceiling)
        self.assertTrue(descriptor.within_probe_ceiling)
        summary = summarize_descriptor(self.artifact, descriptor)
        self.assertFalse(summary["directory"]["within_current_admission_ceiling"])
        self.assertTrue(summary["directory"]["within_probe_ceiling"])
        self.assertFalse(summary["central_directory_fetched"])
        self.assertFalse(summary["member_payload_fetched"])

    def test_descriptor_accepts_observation_inside_current_ceiling(self):
        descriptor = DirectoryDescriptor(2017, 1000, 500_000)
        self.assertTrue(descriptor.within_current_admission_ceiling)
        self.assertTrue(descriptor.within_probe_ceiling)

    def test_descriptor_rejects_zip64_sentinel(self):
        archive_size = 128
        eocd = struct.pack(
            "<4s4H2LH",
            b"PK\x05\x06",
            0,
            0,
            0xFFFF,
            0xFFFF,
            1,
            1,
            0,
        )
        with self.assertRaisesRegex(ValueError, "ZIP64"):
            inspect_directory_descriptor(
                eocd,
                tail_start=archive_size - len(eocd),
                archive_size=archive_size,
            )

    def test_actions_notice_escaping(self):
        self.assertEqual(_github_escape("a%\nb\r"), "a%25%0Ab%0D")


if __name__ == "__main__":
    unittest.main()
