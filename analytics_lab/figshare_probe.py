"""Live, bounded Figshare archive-structure probe for the reviewed fall-video source.

The first live index attempt proved that reviewed metadata and the bounded ZIP
tail are reachable, but the strict central-directory object rejected the remote
EOCD values before exposing which ceiling was exceeded.  This probe therefore
measures only the EOCD descriptor from the same bounded tail.  It deliberately
does not fetch central-directory bytes or member payloads when deciding whether
the existing admission ceiling is suitable.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import PurePosixPath
import struct
from typing import Any

from .figshare_acquisition import (
    FigshareArtifact,
    ZipMember,
    bounded_video_members,
    fetch_article_metadata,
    fetch_exact_range,
)

_MAX_SAMPLE_MEMBERS = 32
_MAX_GROUPS = 24
_TAIL_BYTES = 128 * 1024
_CURRENT_MAX_INDEX_BYTES = 2 * 1024 * 1024
_CURRENT_MAX_ENTRIES = 10_000
_EOCD = b"PK\x05\x06"


@dataclass(frozen=True)
class DirectoryDescriptor:
    """Observed classic-ZIP EOCD values; observation is not admission."""

    entry_count: int
    offset: int
    size: int

    def __post_init__(self) -> None:
        if (type(self.entry_count) is not int or not 0 <= self.entry_count < 0xFFFF
                or type(self.offset) is not int or not 0 <= self.offset < 0xFFFFFFFF
                or type(self.size) is not int or not 0 <= self.size < 0xFFFFFFFF):
            raise ValueError("invalid classic ZIP directory descriptor")

    @property
    def within_current_admission_ceiling(self) -> bool:
        return (
            1 <= self.entry_count <= _CURRENT_MAX_ENTRIES
            and 1 <= self.size <= _CURRENT_MAX_INDEX_BYTES
        )


def inspect_directory_descriptor(
    tail: bytes,
    *,
    tail_start: int,
    archive_size: int,
) -> DirectoryDescriptor:
    """Parse only the EOCD from a bounded tail without fetching the directory."""
    if (not isinstance(tail, (bytes, bytearray)) or type(tail_start) is not int
            or type(archive_size) is not int or archive_size <= 0 or tail_start < 0
            or tail_start + len(tail) != archive_size or len(tail) > _TAIL_BYTES):
        raise ValueError("invalid bounded ZIP tail")
    offset = bytes(tail).rfind(_EOCD)
    if offset < 0 or offset + 22 > len(tail):
        raise ValueError("ZIP end-of-central-directory not found in bounded tail")
    fields = struct.unpack_from("<4s4H2LH", tail, offset)
    (_, disk_no, central_disk, disk_entries, total_entries,
     central_size, central_offset, comment_len) = fields
    if offset + 22 + comment_len != len(tail):
        raise ValueError("ZIP EOCD comment length mismatch")
    if disk_no != 0 or central_disk != 0 or disk_entries != total_entries:
        raise ValueError("multi-disk ZIP archives are not supported")
    if total_entries == 0xFFFF or central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
        raise ValueError("ZIP64 requires a separately reviewed bounded parser")
    descriptor = DirectoryDescriptor(total_entries, central_offset, central_size)
    if descriptor.offset + descriptor.size > archive_size:
        raise ValueError("ZIP central directory extends beyond archive size")
    return descriptor


def fetch_directory_descriptor() -> tuple[FigshareArtifact, DirectoryDescriptor]:
    """Fetch reviewed metadata plus at most 128 KiB of tail; never the directory body."""
    artifact = fetch_article_metadata()
    tail_size = min(_TAIL_BYTES, artifact.size_bytes)
    tail_start = artifact.size_bytes - tail_size
    tail = fetch_exact_range(
        artifact.download_url,
        tail_start,
        artifact.size_bytes - 1,
        total_size=artifact.size_bytes,
    )
    return artifact, inspect_directory_descriptor(
        tail,
        tail_start=tail_start,
        archive_size=artifact.size_bytes,
    )


def summarize_descriptor(
    artifact: FigshareArtifact,
    descriptor: DirectoryDescriptor,
) -> dict[str, Any]:
    if not isinstance(artifact, FigshareArtifact) or not isinstance(descriptor, DirectoryDescriptor):
        raise TypeError("artifact and descriptor types are required")
    return {
        "artifact": {
            "file_id": artifact.file_id,
            "name": artifact.name,
            "size_bytes": artifact.size_bytes,
            "md5": artifact.md5,
        },
        "directory": {
            "entry_count": descriptor.entry_count,
            "offset": descriptor.offset,
            "size": descriptor.size,
            "current_entry_ceiling": _CURRENT_MAX_ENTRIES,
            "current_size_ceiling": _CURRENT_MAX_INDEX_BYTES,
            "within_current_admission_ceiling": descriptor.within_current_admission_ceiling,
        },
        "central_directory_fetched": False,
        "member_payload_fetched": False,
    }


def summarize_index(
    artifact: FigshareArtifact,
    members: tuple[ZipMember, ...],
    *,
    sample_limit: int = _MAX_SAMPLE_MEMBERS,
) -> dict[str, Any]:
    """Return a compact deterministic member summary after a future admitted index fetch."""
    if type(sample_limit) is not int or not 1 <= sample_limit <= _MAX_SAMPLE_MEMBERS:
        raise ValueError("sample_limit outside bounded summary ceiling")
    if not isinstance(artifact, FigshareArtifact):
        raise TypeError("artifact must be FigshareArtifact")
    if not isinstance(members, tuple) or not all(isinstance(item, ZipMember) for item in members):
        raise TypeError("members must be a tuple of ZipMember")

    videos = bounded_video_members(members)
    extensions = Counter(PurePosixPath(item.name).suffix.lower() for item in videos)
    top_levels = Counter(
        PurePosixPath(item.name).parts[0] if PurePosixPath(item.name).parts else ""
        for item in videos
    )
    sample = sorted(videos, key=lambda item: (item.uncompressed_size, item.name))[:sample_limit]
    return {
        "artifact": {
            "file_id": artifact.file_id,
            "name": artifact.name,
            "size_bytes": artifact.size_bytes,
            "md5": artifact.md5,
        },
        "archive_member_count": len(members),
        "bounded_video_count": len(videos),
        "bounded_video_extensions": dict(sorted(extensions.items())),
        "bounded_video_top_levels": [
            {"name": name, "count": count}
            for name, count in sorted(top_levels.items(), key=lambda item: (-item[1], item[0]))[:_MAX_GROUPS]
        ],
        "smallest_bounded_videos": [
            {
                "name": item.name,
                "compressed_size": item.compressed_size,
                "uncompressed_size": item.uncompressed_size,
                "compression_method": item.compression_method,
                "crc32": f"{item.crc32:08x}",
                "local_header_offset": item.local_header_offset,
            }
            for item in sample
        ],
    }


def _github_escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main() -> int:
    artifact, descriptor = fetch_directory_descriptor()
    payload = json.dumps(summarize_descriptor(artifact, descriptor), sort_keys=True, separators=(",", ":"))
    print("figshare_bounded_descriptor=" + payload)
    if os.getenv("GITHUB_ACTIONS") == "true":
        print("::notice title=Figshare bounded descriptor::" + _github_escape(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
