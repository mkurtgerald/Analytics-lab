"""Bounded admission probe for one reviewed CC0 pedestrian tracking clip.

This evidence helper is intentionally narrow: it streams the exact Wikimedia
Commons source on an ``evidence/tracking-cc0-*`` hosted-runner PR, verifies the
published byte length/SHA-1 and the previously discovered SHA-256, and emits
metadata only. It does not retain or upload media, run a detector, or produce
any tracking-accuracy claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import BinaryIO, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SOURCE_PAGE = (
    "https://commons.wikimedia.org/w/index.php?title="
    "File:Video_Codec_Test_pedestrian_area_1080p25.y4m.webm&oldid=1196486240"
)
SOURCE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/a/ae/"
    "Video_Codec_Test_pedestrian_area_1080p25.y4m.webm"
)
RIGHTS_SOURCE = SOURCE_PAGE
LICENSE_EXPRESSION = "CC0-1.0"
AUTHOR = "Taurus Media Technik"
PUBLISHED_SIZE = 11_215_394
PUBLISHED_SHA1 = "51e89a672896e45cca17aa46cd223630a6266e26"
PUBLISHED_SHA256 = "bfadaa62cccb42db875d50bb842aa0964fbf72040432e4097c1df59e043e0c26"
MAX_BYTES = 12_000_000
CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class AdmissionDigest:
    size_bytes: int
    sha1: str
    sha256: str

    def __post_init__(self) -> None:
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("size_bytes must be a positive integer")
        for name in ("sha1", "sha256"):
            value = getattr(self, name)
            expected = 40 if name == "sha1" else 64
            if not isinstance(value, str) or len(value) != expected or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(f"{name} must be lowercase hexadecimal")


def _hash_stream(stream: BinaryIO, *, max_bytes: int = MAX_BYTES) -> AdmissionDigest:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    total = 0
    while True:
        chunk = stream.read(CHUNK_BYTES)
        if not chunk:
            break
        if not isinstance(chunk, (bytes, bytearray)):
            raise ValueError("evidence response must yield bytes")
        total += len(chunk)
        if total > max_bytes:
            raise RuntimeError("evidence asset exceeded bounded download size")
        sha1.update(chunk)
        sha256.update(chunk)
    if total == 0:
        raise RuntimeError("evidence asset was empty")
    return AdmissionDigest(total, sha1.hexdigest(), sha256.hexdigest())


def admit_source(*, opener: Callable[..., BinaryIO] = urlopen) -> AdmissionDigest:
    """Stream and verify the one reviewed source without retaining its bytes."""
    request = Request(
        SOURCE_URL,
        headers={"User-Agent": "Analytics-Lab-evidence-admission/1.0"},
        method="GET",
    )
    with opener(request, timeout=30) as response:
        final_url = response.geturl()
        parsed = urlparse(final_url)
        expected = urlparse(SOURCE_URL)
        if parsed.scheme != "https" or parsed.netloc != expected.netloc or parsed.path != expected.path:
            raise RuntimeError("evidence source redirected outside the reviewed asset")
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                advertised = int(content_length)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("invalid evidence Content-Length") from exc
            if advertised != PUBLISHED_SIZE:
                raise RuntimeError("evidence Content-Length does not match the published asset")
        digest = _hash_stream(response)

    if digest.size_bytes != PUBLISHED_SIZE:
        raise RuntimeError("evidence byte length does not match the published asset")
    if digest.sha1 != PUBLISHED_SHA1:
        raise RuntimeError("evidence SHA-1 does not match the published Wikimedia checksum")
    if digest.sha256 != PUBLISHED_SHA256:
        raise RuntimeError("evidence SHA-256 does not match the pinned admitted asset")
    return digest


def main() -> int:
    digest = admit_source()
    print(json.dumps({
        "source_page": SOURCE_PAGE,
        "rights_source": RIGHTS_SOURCE,
        "source_url": SOURCE_URL,
        "license": LICENSE_EXPRESSION,
        "author": AUTHOR,
        "commercial_evaluation_authorized": True,
        "size_bytes": digest.size_bytes,
        "published_sha1": digest.sha1,
        "pinned_sha256": digest.sha256,
        "media_retained": False,
        "accuracy_claim": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
