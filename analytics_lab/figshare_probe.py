"""Live, bounded Figshare archive-index probe for the reviewed fall-video source.

This command fetches only the reviewed article metadata plus the ZIP tail and
central directory through the fail-closed range adapter.  It never downloads a
member payload and emits only aggregate/public archive-index metadata suitable
for an evidence-branch CI notice.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import PurePosixPath
from typing import Any

from .figshare_acquisition import (
    FigshareArtifact,
    ZipMember,
    bounded_video_members,
    fetch_bounded_archive_index,
)

_MAX_SAMPLE_MEMBERS = 32
_MAX_GROUPS = 24


def summarize_index(
    artifact: FigshareArtifact,
    members: tuple[ZipMember, ...],
    *,
    sample_limit: int = _MAX_SAMPLE_MEMBERS,
) -> dict[str, Any]:
    """Return a compact deterministic summary without touching member bodies."""
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
    artifact, members = fetch_bounded_archive_index()
    summary = summarize_index(artifact, members)
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    print("figshare_bounded_index=" + payload)
    if os.getenv("GITHUB_ACTIONS") == "true":
        print("::notice title=Figshare bounded index::" + _github_escape(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
