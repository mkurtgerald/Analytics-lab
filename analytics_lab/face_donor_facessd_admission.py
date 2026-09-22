"""Admission-only probe for Google's FaceSSD MobileNetV2 Open Images V4 donor.

This module downloads one exact upstream TensorFlow model-zoo archive, bounds the
payload, computes immutable hashes and inspects tar members in memory. It
performs no inference, extracts no files to disk, retains no model artifact and
uses no image/media input.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.parse
import urllib.request

_ARCHIVE_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/"
    "facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz"
)
_ALLOWED_HOST = "storage.googleapis.com"
_MAX_ARCHIVE_BYTES = 130_655_026
_USER_AGENT = "Analytics-Lab-face-donor-admission/1"
_EXPECTED_ARCHIVE_SIZE = 130_655_026
_EXPECTED_ARCHIVE_SHA256 = "9ae49a245caddbe7d7bbc82a35da0191a2f2e210161df19be357a1c7f49118d5"
_EXPECTED_ARCHIVE_MEMBERS: tuple[tuple[str, int, str], ...] | None = None


def _declared_size() -> int | None:
    parsed = urllib.parse.urlparse(_ARCHIVE_URL)
    if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST:
        raise RuntimeError("unapproved face donor URL")
    request = urllib.request.Request(
        _ARCHIVE_URL,
        headers={"User-Agent": _USER_AGENT},
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        final = urllib.parse.urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != _ALLOWED_HOST:
            raise RuntimeError("face donor redirect escaped approved host")
        declared = response.headers.get("Content-Length")
    if declared is None:
        return None
    try:
        value = int(declared)
    except ValueError as exc:
        raise RuntimeError("invalid Content-Length") from exc
    if value < 1:
        raise RuntimeError("invalid Content-Length")
    return value


def _download() -> bytes:
    parsed = urllib.parse.urlparse(_ARCHIVE_URL)
    if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST:
        raise RuntimeError("unapproved face donor URL")
    request = urllib.request.Request(_ARCHIVE_URL, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        final = urllib.parse.urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != _ALLOWED_HOST:
            raise RuntimeError("face donor redirect escaped approved host")
        declared = response.headers.get("Content-Length")
        if declared is not None:
            try:
                declared_size = int(declared)
            except ValueError as exc:
                raise RuntimeError("invalid Content-Length") from exc
            if declared_size != _EXPECTED_ARCHIVE_SIZE:
                raise RuntimeError("face donor archive size mismatch")
        payload = response.read(_MAX_ARCHIVE_BYTES + 1)
    if len(payload) != _EXPECTED_ARCHIVE_SIZE:
        raise RuntimeError("face donor archive size mismatch")
    return payload


def _safe_members(payload: bytes) -> list[dict[str, object]]:
    members: list[dict[str, object]] = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if len(members) >= 128:
                raise RuntimeError("face donor archive member count exceeds bound")
            name = member.name
            if (
                not isinstance(name, str)
                or not name
                or name.startswith("/")
                or "\\" in name
                or ".." in name.split("/")
            ):
                raise RuntimeError("unsafe face donor archive member path")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError("unsafe face donor archive member type")
            if member.isfile() and member.size > _MAX_ARCHIVE_BYTES:
                raise RuntimeError("face donor archive member exceeds size bound")

            item: dict[str, object] = {
                "name": name,
                "size": int(member.size),
                "type": "file" if member.isfile() else "dir" if member.isdir() else "other",
            }
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError("face donor archive member unreadable")
                digest = hashlib.sha256()
                seen = 0
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    seen += len(chunk)
                    if seen > member.size:
                        raise RuntimeError("face donor archive member length mismatch")
                    digest.update(chunk)
                if seen != member.size:
                    raise RuntimeError("face donor archive member length mismatch")
                item["sha256"] = digest.hexdigest()
            members.append(item)
    return members


def _member_identity(members: list[dict[str, object]]) -> tuple[tuple[str, int, str], ...]:
    identity: list[tuple[str, int, str]] = []
    for item in members:
        if item["type"] != "file":
            continue
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("face donor archive member hash missing")
        identity.append((str(item["name"]), int(item["size"]), digest))
    return tuple(identity)


def run() -> dict[str, object]:
    declared = _declared_size()
    if declared is not None and declared != _EXPECTED_ARCHIVE_SIZE:
        raise RuntimeError("face donor archive size mismatch")

    payload = _download()
    sha256 = hashlib.sha256(payload).hexdigest()
    if sha256 != _EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("face donor archive SHA-256 mismatch")

    members = _safe_members(payload)
    identity = _member_identity(members)
    if _EXPECTED_ARCHIVE_MEMBERS is not None and identity != _EXPECTED_ARCHIVE_MEMBERS:
        raise RuntimeError("face donor archive member identity mismatch")

    interesting = [
        item for item in members
        if any(
            str(item["name"]).endswith(suffix)
            for suffix in (".tflite", ".pb", ".config", ".ckpt.index")
        )
    ]
    if not interesting:
        raise RuntimeError("face donor archive contains no recognized model/config member")
    return {
        "evidence": "face-donor-facessd-admission-v2",
        "source_url": _ARCHIVE_URL,
        "archive_bytes": len(payload),
        "archive_sha256": sha256,
        "archive_pinned": True,
        "members_pinned": _EXPECTED_ARCHIVE_MEMBERS is not None,
        "members": members,
        "interesting_members": interesting,
        "total_members": len(members),
        "inference_run": False,
        "media_used": False,
        "claim": "artifact identity/provenance admission only; not face-detection accuracy",
    }


def main() -> int:
    print(json.dumps(run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
