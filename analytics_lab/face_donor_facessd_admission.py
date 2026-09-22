"""Admission-only probe for Google's FaceSSD MobileNetV2 Open Images V4 donor.

This module downloads one exact upstream TensorFlow model-zoo archive, bounds the
payload, computes immutable hashes and inspects tar member metadata in memory.
It performs no inference, extracts no files, retains no model artifact and uses
no image/media input. The first run exists only to discover/pin artifact identity.
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
_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_USER_AGENT = "Analytics-Lab-face-donor-admission/1"
_EXPECTED_ARCHIVE_SIZE: int | None = None
_EXPECTED_ARCHIVE_SHA256: str | None = None


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
            if declared_size < 1 or declared_size > _MAX_ARCHIVE_BYTES:
                raise RuntimeError("face donor archive exceeds size bound")
        payload = response.read(_MAX_ARCHIVE_BYTES + 1)
    if not 1 <= len(payload) <= _MAX_ARCHIVE_BYTES:
        raise RuntimeError("face donor archive exceeds size bound")
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
            members.append(
                {
                    "name": name,
                    "size": int(member.size),
                    "type": "file" if member.isfile() else "dir" if member.isdir() else "other",
                }
            )
    return members


def run() -> dict[str, object]:
    payload = _download()
    size = len(payload)
    sha256 = hashlib.sha256(payload).hexdigest()
    if _EXPECTED_ARCHIVE_SIZE is not None and size != _EXPECTED_ARCHIVE_SIZE:
        raise RuntimeError("face donor archive size mismatch")
    if _EXPECTED_ARCHIVE_SHA256 is not None and sha256 != _EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("face donor archive SHA-256 mismatch")
    members = _safe_members(payload)
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
        "evidence": "face-donor-facessd-admission-v1",
        "source_url": _ARCHIVE_URL,
        "archive_bytes": size,
        "archive_sha256": sha256,
        "pinned": _EXPECTED_ARCHIVE_SIZE is not None and _EXPECTED_ARCHIVE_SHA256 is not None,
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
