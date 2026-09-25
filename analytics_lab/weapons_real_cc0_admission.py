"""Admission-only probe for two rights-cleared real Weapons evidence sources.

This probe streams exactly one CC0 rifle source and one CC0 non-weapon source
from Wikimedia Commons. It verifies their published byte lengths and SHA-1
identities, discovers/pins SHA-256, and emits provenance metadata only. It does
not decode media, construct or execute a model, retain media, or make a
weapon-detection accuracy claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_USER_AGENT = "Analytics-lab bounded Weapons evidence/1.0"
_CHUNK = 64 * 1024


@dataclass(frozen=True)
class _Source:
    name: str
    role: str
    page: str
    url: str
    license: str
    expected_size: int
    expected_sha1: str
    width: int
    height: int
    sha256: str | None = None

    @property
    def max_bytes(self) -> int:
        return self.expected_size + 4096


_RIFLE = _Source(
    name="winchester-model-1895",
    role="weapon-positive-candidate-rifle",
    page=(
        "https://commons.wikimedia.org/w/index.php?title="
        "File:Winchester_Model_1895_Takedown_Rifle_(serial_no._81851),_"
        "custom_built_1913,_right_side.jpg&oldid=1236005629"
    ),
    url=(
        "https://upload.wikimedia.org/wikipedia/commons/2/25/"
        "Winchester_Model_1895_Takedown_Rifle_%28serial_no._81851%29%2C_"
        "custom_built_1913%2C_right_side.jpg"
    ),
    license="CC0-1.0",
    expected_size=1_068_333,
    expected_sha1="a4e50100507987ce58ff128d93eb2b11bccd88e2",
    width=4000,
    height=1663,
)

_CHAIR = _Source(
    name="chair",
    role="nonweapon-negative-candidate",
    page="https://commons.wikimedia.org/w/index.php?title=File:Chair.JPG&oldid=1126064973",
    url="https://upload.wikimedia.org/wikipedia/commons/5/55/Chair.JPG",
    license="CC0-1.0",
    expected_size=2_508_211,
    expected_sha1="f131416b5f2757d57b5a6fdb8b049c7bad87872b",
    width=2848,
    height=4272,
)

_SOURCES = (_RIFLE, _CHAIR)


def _stream_identity(source: _Source, opener: Callable = urlopen) -> tuple[int, str, str]:
    request = Request(source.url, headers={"User-Agent": _USER_AGENT})
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    total = 0
    with opener(request, timeout=20) as response:
        final = urlparse(response.geturl())
        expected = urlparse(source.url)
        if (
            final.scheme != "https"
            or final.netloc != expected.netloc
            or final.path != expected.path
        ):
            raise RuntimeError("Weapons evidence source redirected outside the reviewed asset")
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                advertised = int(content_length)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("invalid Weapons evidence Content-Length") from exc
            if advertised != source.expected_size:
                raise RuntimeError(
                    "Weapons evidence Content-Length does not match the published asset"
                )
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray)):
                raise ValueError("Weapons evidence response must yield bytes")
            total += len(chunk)
            if total > source.max_bytes:
                raise RuntimeError("Weapons evidence source exceeds bounded admission limit")
            sha1.update(chunk)
            sha256.update(chunk)
    if total == 0:
        raise RuntimeError("Weapons evidence source was empty")
    return total, sha1.hexdigest(), sha256.hexdigest()


def _admit(source: _Source, opener: Callable = urlopen) -> dict[str, object]:
    size, sha1, sha256 = _stream_identity(source, opener)
    if size != source.expected_size:
        raise RuntimeError("Weapons evidence source byte length changed")
    if sha1 != source.expected_sha1:
        raise RuntimeError("Weapons evidence source SHA-1 changed")
    if source.sha256 is not None and sha256 != source.sha256:
        raise RuntimeError("Weapons evidence source SHA-256 changed")
    return {
        "name": source.name,
        "role": source.role,
        "source_page": source.page,
        "source_url": source.url,
        "source_license": source.license,
        "source_size": size,
        "source_sha1": sha1,
        "source_sha256": sha256,
        "sha256_pinned": source.sha256 is not None,
        "published_dimensions": [source.width, source.height],
        "media_decoded": False,
        "model_used": False,
        "inference_run": False,
        "artifact_retained": False,
    }


def run(opener: Callable = urlopen) -> dict[str, object]:
    results = [_admit(source, opener) for source in _SOURCES]
    return {
        "evidence": "weapons-real-cc0-source-admission-v1",
        "sources": results,
        "source_count": len(results),
        "all_sha256_pinned": all(bool(item["sha256_pinned"]) for item in results),
        "claim": (
            "source identity/provenance admission only; source roles are evidence "
            "selection metadata, not independently scored detection ground truth"
        ),
    }


def main() -> int:
    print(json.dumps(run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
