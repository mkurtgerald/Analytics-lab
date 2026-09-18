from __future__ import annotations

import io
import json
import unittest
import zipfile

from analytics_lab.figshare_acquisition import (
    FIGSHARE_API_URL,
    FIGSHARE_ARTICLE_ID,
    FIGSHARE_FILE_ID,
    FIGSHARE_VERSION,
    bounded_video_members,
    fetch_bounded_archive_index,
    fetch_exact_range,
    locate_zip_directory,
    parse_article_metadata,
    parse_zip_directory,
)
from analytics_lab.source_rights import FIGSHARE_FALL_2017


class _Response:
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str] | None = None):
        self._stream = io.BytesIO(payload)
        self.status = status
        self.headers = headers or {}

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _archive() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("subject-05/fall-01.mp4", b"fall-bytes")
        archive.writestr("subject-05/adl-01.mp4", b"adl-bytes")
        archive.writestr("notes/readme.txt", b"metadata")
    return stream.getvalue()


def _large_archive() -> bytes:
    """Keep member headers outside the bounded 128 KiB tail used by the index probe."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("subject-05/fall-01.mp4", b"fall-bytes")
        archive.writestr("subject-05/adl-01.mp4", b"adl-bytes")
        archive.writestr("notes/readme.txt", b"metadata")
        archive.writestr(
            "padding.bin",
            b"x" * (192 * 1024),
            compress_type=zipfile.ZIP_STORED,
        )
    return stream.getvalue()


def _metadata(archive: bytes) -> bytes:
    return json.dumps({
        "id": FIGSHARE_ARTICLE_ID,
        "version": FIGSHARE_VERSION,
        "title": FIGSHARE_FALL_2017.title,
        "license": {
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "files": [{
            "id": FIGSHARE_FILE_ID,
            "name": "fall-videos.zip",
            "size": len(archive),
            "download_url": f"https://figshare.com/ndownloader/files/{FIGSHARE_FILE_ID}",
            "computed_md5": "0123456789abcdef0123456789abcdef",
        }],
    }).encode("utf-8")


class FigshareAcquisitionTests(unittest.TestCase):
    def test_metadata_binds_reviewed_article_version_file_and_license(self):
        artifact = parse_article_metadata(_metadata(_archive()))
        self.assertEqual(artifact.file_id, FIGSHARE_FILE_ID)
        self.assertEqual(artifact.name, "fall-videos.zip")
        self.assertTrue(artifact.download_url.startswith("https://"))

    def test_metadata_rejects_wrong_version(self):
        payload = json.loads(_metadata(_archive()).decode("utf-8"))
        payload["version"] = 1
        with self.assertRaisesRegex(ValueError, "article/version"):
            parse_article_metadata(json.dumps(payload).encode("utf-8"))

    def test_bounded_zip_directory_parser_finds_safe_video_members(self):
        archive = _archive()
        directory = locate_zip_directory(archive, tail_start=0, archive_size=len(archive))
        central = archive[directory.offset:directory.offset + directory.size]
        members = parse_zip_directory(central, directory)
        self.assertEqual(directory.entry_count, 3)
        self.assertEqual(len(members), 3)
        videos = bounded_video_members(members)
        self.assertEqual([item.name for item in videos], [
            "subject-05/fall-01.mp4",
            "subject-05/adl-01.mp4",
        ])

    def test_exact_range_fails_closed_if_server_ignores_range(self):
        def opener(request, timeout):
            return _Response(b"entire-body", status=200)

        with self.assertRaisesRegex(RuntimeError, "ignored exact Range"):
            fetch_exact_range(
                f"https://figshare.com/ndownloader/files/{FIGSHARE_FILE_ID}",
                5,
                8,
                total_size=20,
                opener=opener,
            )

    def test_high_level_index_fetch_never_requests_member_payload(self):
        # The first attempt used a sub-128 KiB synthetic archive. By definition,
        # a bounded 128 KiB tail request then covered that entire tiny fixture,
        # including local member headers. Use a larger generated archive so this
        # test exercises the production invariant: only the tail and central
        # directory of a large remote object are requested.
        archive = _large_archive()
        metadata = _metadata(archive)
        requested_ranges: list[tuple[int, int]] = []

        def opener(request, timeout):
            if request.full_url == FIGSHARE_API_URL:
                return _Response(metadata, status=200)
            range_header = request.headers.get("Range")
            self.assertIsNotNone(range_header)
            prefix, bounds = range_header.split("=", 1)
            self.assertEqual(prefix, "bytes")
            start_text, end_text = bounds.split("-", 1)
            start, end = int(start_text), int(end_text)
            requested_ranges.append((start, end))
            payload = archive[start:end + 1]
            return _Response(
                payload,
                status=206,
                headers={"Content-Range": f"bytes {start}-{end}/{len(archive)}"},
            )

        artifact, members = fetch_bounded_archive_index(opener=opener)
        self.assertEqual(artifact.size_bytes, len(archive))
        self.assertEqual(len(members), 4)
        self.assertEqual(len(requested_ranges), 2)
        self.assertTrue(all(end - start + 1 <= 2 * 1024 * 1024 for start, end in requested_ranges))
        self.assertFalse(any(
            start <= member.local_header_offset <= end
            for start, end in requested_ranges
            for member in members
        ))


if __name__ == "__main__":
    unittest.main()
