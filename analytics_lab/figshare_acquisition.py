"""Bounded metadata/index access for the reviewed Figshare fall-video source.

This module deliberately does *not* download video members.  It closes the
safe acquisition boundary needed before broader-source evaluation by allowing
an exact reviewed Figshare article/file to be identified and a ZIP central
index to be inspected with small HTTP Range requests.  All payload limits are
hard ceilings and any server that ignores Range fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import PurePosixPath
import re
import struct
from typing import Any, Callable
import urllib.request

from .source_rights import FIGSHARE_FALL_2017, require_source

FIGSHARE_ARTICLE_ID = 28_596_332
FIGSHARE_VERSION = 2
FIGSHARE_FILE_ID = 52_990_358
FIGSHARE_API_URL = (
    f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/versions/{FIGSHARE_VERSION}"
)

_MAX_METADATA_BYTES = 512 * 1024
_MAX_TAIL_BYTES = 128 * 1024
_MAX_INDEX_BYTES = 2 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
_MD5 = re.compile(r"[0-9a-f]{32}\Z")
_HTTPS = re.compile(r"https://[^\s]+\Z")
_EOCD = b"PK\x05\x06"
_CENTRAL = b"PK\x01\x02"


@dataclass(frozen=True)
class FigshareArtifact:
    file_id: int
    name: str
    size_bytes: int
    download_url: str
    md5: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.name)
        if (type(self.file_id) is not int or self.file_id <= 0 or not self.name
                or len(self.name) > 256 or path.is_absolute() or ".." in path.parts
                or "\\" in self.name):
            raise ValueError("invalid Figshare file identity")
        if type(self.size_bytes) is not int or not 1 <= self.size_bytes <= _MAX_ARCHIVE_BYTES:
            raise ValueError("Figshare archive size is outside the reviewed bound")
        if not isinstance(self.download_url, str) or not _HTTPS.fullmatch(self.download_url):
            raise ValueError("Figshare download URL must be bounded HTTPS metadata")
        if not isinstance(self.md5, str) or not _MD5.fullmatch(self.md5):
            raise ValueError("Figshare file metadata must include lowercase MD5")


@dataclass(frozen=True)
class ZipDirectory:
    entry_count: int
    offset: int
    size: int

    def __post_init__(self) -> None:
        if (type(self.entry_count) is not int or not 1 <= self.entry_count <= 10_000
                or type(self.offset) is not int or self.offset < 0
                or type(self.size) is not int or not 1 <= self.size <= _MAX_INDEX_BYTES):
            raise ValueError("ZIP directory exceeds bounded index limits")


@dataclass(frozen=True)
class ZipMember:
    name: str
    compressed_size: int
    uncompressed_size: int
    compression_method: int
    crc32: int
    local_header_offset: int

    @property
    def is_bounded_video(self) -> bool:
        suffix = PurePosixPath(self.name).suffix.lower()
        return (
            suffix in {".mp4", ".avi", ".mov", ".mkv"}
            and 0 < self.compressed_size <= MAX_MEMBER_BYTES
            and 0 < self.uncompressed_size <= MAX_MEMBER_BYTES
        )


def _read_bounded(response: Any, limit: int) -> bytes:
    if not hasattr(response, "read"):
        raise RuntimeError("HTTP response is not readable")
    data = response.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError("HTTP response exceeded the bounded byte ceiling")
    return data


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if type(status) is int:
        return status
    if hasattr(response, "getcode"):
        code = response.getcode()
        if type(code) is int:
            return code
    raise RuntimeError("HTTP response status unavailable")


def parse_article_metadata(payload: bytes) -> FigshareArtifact:
    """Validate exact reviewed article/version/file metadata from the Figshare API."""
    require_source(FIGSHARE_FALL_2017.source_id, purpose="evaluation", require_real_world=True)
    if not isinstance(payload, (bytes, bytearray)) or len(payload) > _MAX_METADATA_BYTES:
        raise ValueError("invalid bounded Figshare metadata payload")
    try:
        article = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Figshare metadata JSON") from exc
    if not isinstance(article, dict):
        raise ValueError("Figshare article metadata must be an object")
    if article.get("id") != FIGSHARE_ARTICLE_ID or article.get("version") != FIGSHARE_VERSION:
        raise ValueError("Figshare article/version identity mismatch")
    if article.get("title") != FIGSHARE_FALL_2017.title:
        raise ValueError("Figshare article title mismatch")

    license_info = article.get("license")
    if not isinstance(license_info, dict):
        raise ValueError("Figshare license metadata missing")
    license_name = license_info.get("name") or license_info.get("title")
    license_url = license_info.get("url")
    if license_name not in {"CC BY 4.0", "CC-BY-4.0", "Creative Commons Attribution 4.0 International"}:
        raise ValueError("Figshare license name mismatch")
    if license_url is not None and "creativecommons.org/licenses/by/4.0" not in str(license_url):
        raise ValueError("Figshare license URL mismatch")

    files = article.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Figshare file metadata missing")
    matches = [item for item in files if isinstance(item, dict) and item.get("id") == FIGSHARE_FILE_ID]
    if len(matches) != 1:
        raise ValueError("reviewed Figshare file identity not unique")
    item = matches[0]
    md5 = item.get("computed_md5") or item.get("supplied_md5")
    return FigshareArtifact(
        file_id=item.get("id"),
        name=item.get("name"),
        size_bytes=item.get("size"),
        download_url=item.get("download_url"),
        md5=md5,
    )


def fetch_article_metadata(*, opener: Callable[..., Any] = urllib.request.urlopen) -> FigshareArtifact:
    """Fetch only bounded JSON metadata; never media bytes."""
    request = urllib.request.Request(
        FIGSHARE_API_URL,
        headers={"User-Agent": "Analytics-lab-figshare-boundary/1"},
    )
    with opener(request, timeout=20) as response:
        if _response_status(response) != 200:
            raise RuntimeError("Figshare metadata request failed")
        payload = _read_bounded(response, _MAX_METADATA_BYTES)
    return parse_article_metadata(payload)


def fetch_exact_range(
    url: str,
    start: int,
    end: int,
    *,
    total_size: int,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> bytes:
    """Fetch one exact byte range and fail if the server falls back to full-body 200."""
    if (not isinstance(url, str) or not _HTTPS.fullmatch(url)
            or type(start) is not int or type(end) is not int or type(total_size) is not int
            or start < 0 or end < start or end >= total_size):
        raise ValueError("invalid bounded range request")
    expected = end - start + 1
    if expected > _MAX_INDEX_BYTES:
        raise ValueError("range exceeds bounded index budget")
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "Analytics-lab-figshare-boundary/1",
        },
    )
    with opener(request, timeout=20) as response:
        if _response_status(response) != 206:
            raise RuntimeError("remote source ignored exact Range request")
        headers = getattr(response, "headers", {})
        content_range = headers.get("Content-Range") if hasattr(headers, "get") else None
        if content_range != f"bytes {start}-{end}/{total_size}":
            raise RuntimeError("remote Content-Range did not match pinned request")
        payload = _read_bounded(response, expected)
    if len(payload) != expected:
        raise RuntimeError("remote range byte count mismatch")
    return payload


def locate_zip_directory(tail: bytes, *, tail_start: int, archive_size: int) -> ZipDirectory:
    """Locate a classic single-disk ZIP central directory from a bounded archive tail."""
    if (not isinstance(tail, (bytes, bytearray)) or type(tail_start) is not int
            or type(archive_size) is not int or tail_start < 0 or archive_size <= 0
            or tail_start + len(tail) != archive_size or len(tail) > _MAX_TAIL_BYTES):
        raise ValueError("invalid bounded ZIP tail")
    offset = bytes(tail).rfind(_EOCD)
    if offset < 0 or offset + 22 > len(tail):
        raise ValueError("ZIP end-of-central-directory not found in bounded tail")
    fields = struct.unpack_from("<4s4H2LH", tail, offset)
    _, disk_no, central_disk, disk_entries, total_entries, central_size, central_offset, comment_len = fields
    if offset + 22 + comment_len != len(tail):
        raise ValueError("ZIP EOCD comment length mismatch")
    if disk_no != 0 or central_disk != 0 or disk_entries != total_entries:
        raise ValueError("multi-disk ZIP archives are not supported")
    if total_entries == 0xFFFF or central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
        raise ValueError("ZIP64 requires a separately reviewed bounded parser")
    directory = ZipDirectory(total_entries, central_offset, central_size)
    if directory.offset + directory.size > archive_size:
        raise ValueError("ZIP central directory extends beyond archive size")
    return directory


def parse_zip_directory(payload: bytes, directory: ZipDirectory) -> tuple[ZipMember, ...]:
    """Parse a bounded classic ZIP central directory without touching member payloads."""
    if not isinstance(payload, (bytes, bytearray)) or len(payload) != directory.size:
        raise ValueError("central-directory byte count mismatch")
    data = bytes(payload)
    members: list[ZipMember] = []
    cursor = 0
    while cursor < len(data):
        if cursor + 46 > len(data) or data[cursor:cursor + 4] != _CENTRAL:
            raise ValueError("invalid ZIP central-directory record")
        fields = struct.unpack_from("<4s6H3L5H2L", data, cursor)
        (_, _made, _needed, flags, method, _mtime, _mdate, crc32,
         compressed, uncompressed, name_len, extra_len, comment_len,
         disk_start, _internal_attr, _external_attr, local_offset) = fields
        if flags & 0x1:
            raise ValueError("encrypted ZIP members are not supported")
        if disk_start != 0:
            raise ValueError("multi-disk ZIP member is not supported")
        if compressed == 0xFFFFFFFF or uncompressed == 0xFFFFFFFF or local_offset == 0xFFFFFFFF:
            raise ValueError("ZIP64 member requires separate review")
        record_end = cursor + 46 + name_len + extra_len + comment_len
        if record_end > len(data):
            raise ValueError("truncated ZIP central-directory record")
        name_bytes = data[cursor + 46:cursor + 46 + name_len]
        try:
            name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid ZIP member name encoding") from exc
        path = PurePosixPath(name)
        if (not name or len(name) > 512 or path.is_absolute() or ".." in path.parts
                or "\\" in name):
            raise ValueError("unsafe ZIP member path")
        if not name.endswith("/"):
            members.append(ZipMember(name, compressed, uncompressed, method, crc32, local_offset))
        cursor = record_end
    if len(members) > directory.entry_count:
        raise ValueError("ZIP member count exceeds EOCD count")
    return tuple(members)


def fetch_bounded_archive_index(
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[FigshareArtifact, tuple[ZipMember, ...]]:
    """Fetch metadata + bounded ZIP index only; never fetch video member bodies."""
    artifact = fetch_article_metadata(opener=opener)
    tail_size = min(_MAX_TAIL_BYTES, artifact.size_bytes)
    tail_start = artifact.size_bytes - tail_size
    tail = fetch_exact_range(
        artifact.download_url,
        tail_start,
        artifact.size_bytes - 1,
        total_size=artifact.size_bytes,
        opener=opener,
    )
    directory = locate_zip_directory(tail, tail_start=tail_start, archive_size=artifact.size_bytes)
    central = fetch_exact_range(
        artifact.download_url,
        directory.offset,
        directory.offset + directory.size - 1,
        total_size=artifact.size_bytes,
        opener=opener,
    )
    return artifact, parse_zip_directory(central, directory)


def bounded_video_members(members: tuple[ZipMember, ...]) -> tuple[ZipMember, ...]:
    """Return only video members already inside the repository's 16 MiB item ceiling."""
    if not isinstance(members, tuple) or not all(isinstance(item, ZipMember) for item in members):
        raise ValueError("members must be a tuple of ZipMember records")
    return tuple(item for item in members if item.is_bounded_video)
