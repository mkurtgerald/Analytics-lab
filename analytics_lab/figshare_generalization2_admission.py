"""Admit the predeclared second Figshare generalization pair through exact ranges.

The pair was selected from central-directory metadata in PR #50 before any
member payload or model output was inspected. This module pins those measured
archive identities, reuses the reviewed bounded ZIP extraction path, computes
SHA-256 over the uncompressed video bytes, and writes only to caller-selected
ephemeral storage. It does not change analytics thresholds or claim accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .figshare_acquisition import ZipMember
from .figshare_member_admission import AdmittedMember, PinnedMember, fetch_verified_member
from .figshare_probe import fetch_bounded_probe_index
from .source_rights import FIGSHARE_FALL_2017, require_source

PAIR_ID = "generalization-2"

ADL_NEGATIVE = PinnedMember(
    role="negative",
    name="VideoDataset/ADL/SBJ_06_LOC1/ACT19_R_1/20240920140827.mp4",
    compressed_size=592_355,
    uncompressed_size=593_051,
    crc32=0x78D8749B,
    local_header_offset=1_323_417_041,
)
FALL_POSITIVE = PinnedMember(
    role="positive",
    name="VideoDataset/Fall/SBJ_03_LOC2/ACT4_R_1/20240912_111106.mp4",
    compressed_size=598_064,
    uncompressed_size=598_720,
    crc32=0xC8E45DF7,
    local_header_offset=2_000_888_480,
)
PINNED_PAIR = (ADL_NEGATIVE, FALL_POSITIVE)


def select_pinned_pair(members: tuple[ZipMember, ...]) -> tuple[tuple[PinnedMember, ZipMember], ...]:
    if not isinstance(members, tuple) or not all(isinstance(item, ZipMember) for item in members):
        raise TypeError("members must be a tuple of ZipMember")
    selected: list[tuple[PinnedMember, ZipMember]] = []
    for expected in PINNED_PAIR:
        matches = [item for item in members if expected.matches(item)]
        if len(matches) != 1:
            raise RuntimeError("second Figshare member identity is missing or changed")
        member = matches[0]
        if not member.is_bounded_video:
            raise RuntimeError("second Figshare member is outside the reviewed extraction envelope")
        selected.append((expected, member))
    if selected[0][1].name == selected[1][1].name:
        raise RuntimeError("positive and negative members must be file-disjoint")
    return tuple(selected)


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
        filename = (
            "figshare-g2-adl-negative.mp4"
            if expected.role == "negative"
            else "figshare-g2-fall-positive.mp4"
        )
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
        "pair_id": PAIR_ID,
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
    print("figshare_generalization2_member_admission=" + payload)
    if os.getenv("GITHUB_ACTIONS") == "true":
        print("::notice title=Figshare second-pair admission::" + _github_escape(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
