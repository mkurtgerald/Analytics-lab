from __future__ import annotations

import unittest

from analytics_lab.figshare_acquisition import FigshareArtifact, ZipMember
from analytics_lab.figshare_probe import _github_escape, summarize_index


class FigshareProbeTests(unittest.TestCase):
    def setUp(self):
        self.artifact = FigshareArtifact(
            file_id=52990358,
            name="VideoDataset.zip",
            size_bytes=2_000_000,
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

    def test_actions_notice_escaping(self):
        self.assertEqual(_github_escape("a%\nb\r"), "a%25%0Ab%0D")


if __name__ == "__main__":
    unittest.main()
