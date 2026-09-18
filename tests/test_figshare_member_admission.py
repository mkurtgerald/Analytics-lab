from __future__ import annotations

import io
import unittest
import zipfile

from analytics_lab.figshare_acquisition import FigshareArtifact, ZipMember, locate_zip_directory, parse_zip_directory
from analytics_lab.figshare_member_admission import (
    ADL_NEGATIVE,
    FALL_POSITIVE,
    _inflate_member,
    evidence_document,
    fetch_verified_member,
    select_pinned_pair,
    AdmittedMember,
)


class _Response:
    def __init__(self, payload: bytes, *, status: int, content_range: str):
        self._stream = io.BytesIO(payload)
        self.status = status
        self.headers = {"Content-Range": content_range}

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _single_member_archive(payload: bytes) -> tuple[bytes, ZipMember]:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("safe/member.mp4", payload)
    raw = stream.getvalue()
    directory = locate_zip_directory(raw, tail_start=0, archive_size=len(raw))
    members = parse_zip_directory(raw[directory.offset:directory.offset + directory.size], directory)
    return raw, members[0]


class FigshareMemberAdmissionTests(unittest.TestCase):
    def test_selects_only_exact_measured_pair(self):
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
            ZipMember("other.mp4", 10, 12, 8, 1, 10),
        )
        selected = select_pinned_pair(members)
        self.assertEqual([expected.role for expected, _ in selected], ["negative", "positive"])
        self.assertEqual([member.name for _, member in selected], [ADL_NEGATIVE.name, FALL_POSITIVE.name])

    def test_selection_fails_closed_on_changed_identity(self):
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

    def test_fetch_verified_member_uses_exact_ranges_and_hashes_uncompressed_bytes(self):
        source = b"bounded-video-bytes" * 200
        archive, member = _single_member_archive(source)
        artifact = FigshareArtifact(
            file_id=52990358,
            name="fixture.zip",
            size_bytes=len(archive),
            download_url="https://figshare.com/ndownloader/files/52990358",
            md5="0123456789abcdef0123456789abcdef",
        )
        requested: list[tuple[int, int]] = []

        def opener(request, timeout):
            header = request.headers.get("Range")
            self.assertIsNotNone(header)
            start_text, end_text = header.split("=", 1)[1].split("-", 1)
            start, end = int(start_text), int(end_text)
            requested.append((start, end))
            return _Response(
                archive[start:end + 1],
                status=206,
                content_range=f"bytes {start}-{end}/{len(archive)}",
            )

        payload, digest = fetch_verified_member(artifact, member, opener=opener)
        self.assertEqual(payload, source)
        self.assertEqual(len(digest), 64)
        self.assertEqual(len(requested), 3)
        self.assertEqual(requested[0], (member.local_header_offset, member.local_header_offset + 29))
        self.assertEqual(requested[-1][1] - requested[-1][0] + 1, member.compressed_size)

    def test_inflate_rejects_crc_mismatch(self):
        source = b"crc-test" * 50
        archive, member = _single_member_archive(source)
        directory = locate_zip_directory(archive, tail_start=0, archive_size=len(archive))
        local_start = member.local_header_offset
        name_len = int.from_bytes(archive[local_start + 26:local_start + 28], "little")
        extra_len = int.from_bytes(archive[local_start + 28:local_start + 30], "little")
        data_start = local_start + 30 + name_len + extra_len
        compressed = archive[data_start:data_start + member.compressed_size]
        changed = ZipMember(
            member.name,
            member.compressed_size,
            member.uncompressed_size,
            member.compression_method,
            member.crc32 ^ 1,
            member.local_header_offset,
        )
        with self.assertRaisesRegex(RuntimeError, "CRC32"):
            _inflate_member(changed, compressed)

    def test_evidence_document_excludes_local_paths_and_disclaims_accuracy(self):
        admitted = (
            AdmittedMember("negative", ADL_NEGATIVE.name, 8, 10, 20, 1, "a" * 64, __import__("pathlib").Path("/tmp/a.mp4")),
            AdmittedMember("positive", FALL_POSITIVE.name, 8, 11, 21, 2, "b" * 64, __import__("pathlib").Path("/tmp/b.mp4")),
        )
        document = evidence_document(admitted)
        self.assertEqual(document["source"]["license_id"], "CC-BY-4.0")
        self.assertTrue(document["source"]["attribution_required"])
        self.assertFalse(document["commercial_accuracy_claim"])
        self.assertNotIn("local_path", str(document))


if __name__ == "__main__":
    unittest.main()
