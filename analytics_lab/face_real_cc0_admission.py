"""Bounded rights/source admission probe for one real-person CC0 face image.

This probe streams one exact Wikimedia Commons source into memory and verifies
its pinned byte length, SHA-1 and SHA-256. It does not decode the image,
construct or execute a model, retain media, or make a face-detection accuracy
claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Callable
from urllib.request import Request, urlopen

_SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Face_portrait_(Unsplash).jpg"
_SOURCE_URL = "https://upload.wikimedia.org/wikipedia/commons/0/04/Face_portrait_%28Unsplash%29.jpg"
_SOURCE_LICENSE = "CC0-1.0"
_SOURCE_AUTHOR = "William Stitt"
_SOURCE_DATE = "2016-10-19"
_EXPECTED_SIZE = 9_023_019
_EXPECTED_SHA1 = "d7ac58077d135b823e71e7b38e763f082e2053bc"
_SOURCE_SHA256 = "7356daa8fd4ad53b946ce0036f06b014431dc89b7ae29ecd8ef18fc54edce6b5"
_MAX_BYTES = 10_000_000
_CHUNK = 64 * 1024
_USER_AGENT = "Analytics-lab bounded evidence/1.0"


def _stream_identity(opener: Callable = urlopen) -> tuple[int, str, str]:
    request = Request(_SOURCE_URL, headers={"User-Agent": _USER_AGENT})
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    total = 0
    with opener(request, timeout=20) as response:
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_BYTES:
                raise RuntimeError("face source exceeds bounded admission limit")
            sha1.update(chunk)
            sha256.update(chunk)
    return total, sha1.hexdigest(), sha256.hexdigest()


def run(opener: Callable = urlopen) -> dict[str, object]:
    size, sha1, sha256 = _stream_identity(opener)
    if size != _EXPECTED_SIZE:
        raise RuntimeError("face source byte length changed")
    if sha1 != _EXPECTED_SHA1:
        raise RuntimeError("face source SHA-1 changed")
    if sha256 != _SOURCE_SHA256:
        raise RuntimeError("face source SHA-256 changed")
    return {
        "evidence": "face-real-cc0-source-admission-v1",
        "source_page": _SOURCE_PAGE,
        "source_url": _SOURCE_URL,
        "source_license": _SOURCE_LICENSE,
        "source_author": _SOURCE_AUTHOR,
        "source_date": _SOURCE_DATE,
        "source_size": size,
        "source_sha1": sha1,
        "source_sha256": sha256,
        "sha256_pinned": True,
        "media_decoded": False,
        "model_used": False,
        "inference_run": False,
        "artifact_retained": False,
        "claim": "source identity/provenance admission only; no detection accuracy claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
