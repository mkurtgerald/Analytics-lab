"""Admit exactly two reviewed Figshare video members through bounded byte ranges.

This module is intentionally narrower than a general ZIP extractor. It reuses the
reviewed Figshare archive index, requires the exact positive/negative member
identities measured in PR #47, fetches only local-header/name/data ranges, and
verifies ZIP metadata, uncompressed size, CRC32 and SHA-256 before writing an
ordinary local file. The caller chooses an ephemeral output directory; this
module never uploads or retains media in GitHub.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any
import zlib

from .figshare_acquisition import FigshareArtifact, MAX_MEMBER_BYTES, ZipMember, fetch_exact_range
from .figshare_probe import fetch_bounded_probe_index
from .source_rights import FIGSHARE_FALL_2017, require_source

_LOCAL = b"PK\x03\x04"
_LOCAL_FIXED_BYTES = 30
_MAX_LOCAL_EXTRA_BYTES = 4096
_SUPPORTED_METHODS = frozenset({0, 8})
_ALLOWED_FLAGS = 0x0006 | 0x0008 | 0x0800


@dataclass(frozen=True)
class PinnedMember:
    role: str
    name: str
    compressed_size: int
    uncompressed_size: int
    crc32: int
    local_header_offset: int

    def matches(self, member: ZipMember) -> bool:
        return (
            isinstance(member, ZipMember)
            and member.name == self.name
            and member.compressed_size == self.compressed_size
            and member.uncompressed_size == self.uncompressed_size
            and member.crc32 == self.crc32
            and member.local_header_offset == self.local_header_offset
        )


ADL_NEGATIVE = PinnedMember(
    role="negative",
    name="VideoDataset/ADL/SBJ_01_LOC3/ACT25_R_1/20240923130459.mp4",
    compressed_size=277_756,
    uncompressed_size=278_497,
    crc32=0x44AC1304,
    local_header_offset=1_084_380_237,
)
FALL_POSITIVE = PinnedMember(
    role="positive",
    name="VideoDataset/Fall/SBJ_10_LOC3/ACT10_R_2/20240915184434.mp4",
    compressed_size=359_774,
    uncompressed_size=360_498,
    crc32=0x41961303,
    local_header_offset=1_738_471_658,
)
PINNED_PAIR = (ADL_NEGATIVE, FALL_POSITIVE)


@dataclass(frozen=True)
class AdmittedMember:
    role: str
    name: str
    compression_method: int
    compressed_size: int
    uncompressed_size: int
    crc32: int
    sha256: str
    local_path: Path


def select_pinned_pair(members: tuple[ZipMember, ...]) -> tuple[tuple[PinnedMember, ZipMember], ...]:
    if not isinstance(members, tuple) or not all(isinstance(item, ZipMember) for item in members):
        raise TypeError("members must be a tuple of ZipMember")
    selected: list[tuple[PinnedMember, ZipMember]] = []
    for expected in PINNED_PAIR:
        matches = [item for item in members if expected.matches(item)]
        if len(matches) != 1:
            raise RuntimeError("pinned Figshare member identity is missing or changed")
        member = matches[0]
        if not member.is_bounded_video or member.compression_method not in _SUPPORTED_METHODS:
            raise RuntimeError("pinned Figshare member is outside the reviewed extraction envelope")
        selected.append((expected, member))
    if selected[0][1].name == selected[1][1].name:
        raise RuntimeError("positive and negative members must be file-disjoint")
    return tuple(selected)


def _local_data_start(
    artifact: FigshareArtifact,
    member: ZipMember,
    *,
    opener: Any,
) -> int:
    start = member.local_header_offset
    fixed = fetch_exact_range(
        artifact.download_url,
        start,
        start + _LOCAL_FIXED_BYTES - 1,
        total_size=artifact.size_bytes,
        opener=opener,
    )
    if len(fixed) != _LOCAL_FIXED_BYTES or fixed[:4] != _LOCAL:
        raise RuntimeError("ZIP local header signature mismatch")
    (
        _signature,
        _version_needed,
        flags,
        method,
        _mtime,
        _mdate,
        local_crc32,
        local_compressed,
        local_uncompressed,
        name_len,
        extra_len,
    ) = struct.unpack("<4s5H3L2H", fixed)
    if flags & 0x1 or flags & ~_ALLOWED_FLAGS:
        raise RuntimeError("ZIP local member flags are outside the reviewed envelope")
    if method != member.compression_method or method not in _SUPPORTED_METHODS:
        raise RuntimeError("ZIP local compression method mismatch")
    if name_len < 1 or name_len > 512 or extra_len > _MAX_LOCAL_EXTRA_BYTES:
        raise RuntimeError("ZIP local name/extra length exceeds the reviewed envelope")
    if not (flags & 0x8):
        if (
            local_crc32 != member.crc32
            or local_compressed != member.compressed_size
            or local_uncompressed != member.uncompressed_size
        ):
            raise RuntimeError("ZIP local member sizes/CRC do not match the central directory")

    tail_start = start + _LOCAL_FIXED_BYTES
    tail_end = tail_start + name_len + extra_len - 1
    if tail_end >= artifact.size_bytes:
        raise RuntimeError("ZIP local header extends beyond archive size")
    name_and_extra = fetch_exact_range(
        artifact.download_url,
        tail_start,
        tail_end,
        total_size=artifact.size_bytes,
        opener=opener,
    )
    name_bytes = name_and_extra[:name_len]
    try:
        decoded_name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
    except UnicodeDecodeError as exc:
        raise RuntimeError("ZIP local member name encoding mismatch") from exc
    if decoded_name != member.name:
        raise RuntimeError("ZIP local member name does not match the central directory")
    data_start = tail_end + 1
    data_end = data_start + member.compressed_size - 1
    if data_start <= start or data_end >= artifact.size_bytes:
        raise RuntimeError("ZIP member payload range exceeds archive size")
    return data_start


def _inflate_member(member: ZipMember, compressed: bytes) -> bytes:
    if not isinstance(compressed, (bytes, bytearray)) or len(compressed) != member.compressed_size:
        raise RuntimeError("ZIP compressed payload byte count mismatch")
    if member.compression_method == 0:
        data = bytes(compressed)
    elif member.compression_method == 8:
        decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        data = decoder.decompress(bytes(compressed), member.uncompressed_size + 1)
        if len(data) > member.uncompressed_size or decoder.unconsumed_tail:
            raise RuntimeError("ZIP member expanded beyond the declared byte ceiling")
        remaining = member.uncompressed_size + 1 - len(data)
        data += decoder.flush(max(1, remaining))
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise RuntimeError("ZIP deflate stream did not end cleanly")
    else:
        raise RuntimeError("ZIP compression method is not reviewed")
    if len(data) != member.uncompressed_size or len(data) > MAX_MEMBER_BYTES:
        raise RuntimeError("ZIP uncompressed member size mismatch")
    if (zlib.crc32(data) & 0xFFFFFFFF) != member.crc32:
        raise RuntimeError("ZIP member CRC32 mismatch")
    return data


def fetch_verified_member(
    artifact: FigshareArtifact,
    member: ZipMember,
    *,
    opener: Any,
) -> tuple[bytes, str]:
    if not isinstance(artifact, FigshareArtifact) or not isinstance(member, ZipMember):
        raise TypeError("artifact and member are required")
    if not member.is_bounded_video or member.compression_method not in _SUPPORTED_METHODS:
        raise RuntimeError("member is outside the reviewed extraction envelope")
    data_start = _local_data_start(artifact, member, opener=opener)
    compressed = fetch_exact_range(
        artifact.download_url,
        data_start,
        data_start + member.compressed_size - 1,
        total_size=artifact.size_bytes,
        opener=opener,
    )
    data = _inflate_member(member, compressed)
    return data, hashlib.sha256(data).hexdigest()


def admit_pinned_pair(output_dir: str | Path, *, opener: Any) -> tuple[AdmittedMember, ...]:
    rights = require_source(
        FIGSHARE_FALL_2017.source_id,
        purpose="evaluation",
        require_real_world=True,
    )
    if rights is not FIGSHARE_FALL_2017:
        raise RuntimeError("reviewed Figshare rights identity changed")
    root = Path(output_dir)
    if root.exists() and root.is_symlink():
        raise RuntimeError("output directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise RuntimeError("output path must be a directory")

    artifact, _descriptor, members, _ranges = fetch_bounded_probe_index()
    selected = select_pinned_pair(members)
    admitted: list[AdmittedMember] = []
    for expected, member in selected:
        data, sha256 = fetch_verified_member(artifact, member, opener=opener)
        filename = "figshare-adl-negative.mp4" if expected.role == "negative" else "figshare-fall-positive.mp4"
        target = root / filename
        if target.exists() or target.is_symlink():
            raise RuntimeError("refusing to overwrite admitted media")
        target.write_bytes(data)
        if hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
            raise RuntimeError("written admitted media failed SHA-256 verification")
        admitted.append(AdmittedMember(
            role=expected.role,
            name=member.name,
            compression_method=member.compression_method,
            compressed_size=member.compressed_size,
            uncompressed_size=member.uncompressed_size,
            crc32=member.crc32,
            sha256=sha256,
            local_path=target,
        ))
    return tuple(admitted)


def evidence_document(admitted: tuple[AdmittedMember, ...]) -> dict[str, Any]:
    if not isinstance(admitted, tuple) or len(admitted) != 2 or any(not isinstance(item, AdmittedMember) for item in admitted):
        raise TypeError("admitted must be the exact two-member evidence pair")
    return {
        "source": {
            "source_id": FIGSHARE_FALL_2017.source_id,
            "title": FIGSHARE_FALL_2017.title,
            "version": FIGSHARE_FALL_2017.version,
            "license_id": FIGSHARE_FALL_2017.license_id,
            "provenance_ref": FIGSHARE_FALL_2017.provenance_ref,
            "attribution_required": FIGSHARE_FALL_2017.attribution_required,
        },
        "members": [
            {
                "role": item.role,
                "name": item.name,
                "compression_method": item.compression_method,
                "compressed_size": item.compressed_size,
                "uncompressed_size": item.uncompressed_size,
                "crc32": f"{item.crc32:08x}",
                "sha256": item.sha256,
            }
            for item in admitted
        ],
        "member_payload_fetched": True,
        "media_retained_in_repository": False,
        "commercial_accuracy_claim": False,
    }


def _github_escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    import urllib.request

    admitted = admit_pinned_pair(args.output_dir, opener=urllib.request.urlopen)
    payload = json.dumps(evidence_document(admitted), sort_keys=True, separators=(",", ":"))
    print("figshare_member_admission=" + payload)
    if os.getenv("GITHUB_ACTIONS") == "true":
        print("::notice title=Figshare member admission::" + _github_escape(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
