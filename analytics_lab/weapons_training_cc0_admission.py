"""Bounded source admission for first project-owned Weapons training candidates.

This lane streams exactly three uploader-owned CC0 Wikimedia Commons object
images selected for handgun, sword, and non-kitchen-knife coverage. It verifies
the published byte count and SHA-1, discovers or re-verifies SHA-256, and emits
rights/source metadata only.

It does not decode pixels, author boxes, admit an asset into a training split,
install a model/runtime, train, infer, retain media, or make an accuracy claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_USER_AGENT = "Analytics-lab bounded Weapons training CC0 admission/1.0"
_CHUNK = 64 * 1024


@dataclass(frozen=True)
class _Source:
    name: str
    source_class: str
    page: str
    url: str
    license: str
    rights_basis: str
    expected_size: int
    expected_sha1: str
    width: int
    height: int
    identifiable_people: bool
    third_party_caveat: str
    sha256: str | None = None

    @property
    def max_bytes(self) -> int:
        return self.expected_size


_WALTHER = _Source(
    name="walther-p38-pistol",
    source_class="handgun",
    page=(
        "https://commons.wikimedia.org/w/index.php?"
        "title=File:Walther_P.38_pistol.jpg&oldid=1222094101"
    ),
    url="https://upload.wikimedia.org/wikipedia/commons/9/95/Walther_P.38_pistol.jpg",
    license="CC0-1.0",
    rights_basis="uploader-owned own work dedicated to CC0-1.0",
    expected_size=239_858,
    expected_sha1="019a9a8254e967438f0fe84428ecbc4e3ce2cfa3",
    width=1500,
    height=1659,
    identifiable_people=False,
    third_party_caveat=(
        "branded firearm; trademark/non-endorsement review remains before training"
    ),
)

_SWORD = _Source(
    name="british-styled-celtic-la-tene-sword",
    source_class="sword",
    page=(
        "https://commons.wikimedia.org/w/index.php?"
        "title=File:British-styled_Celtic_La_T%C3%A8ne_sword.jpg&oldid=1098660580"
    ),
    url=(
        "https://upload.wikimedia.org/wikipedia/commons/2/2f/"
        "British-styled_Celtic_La_T%C3%A8ne_sword.jpg"
    ),
    license="CC0-1.0",
    rights_basis="uploader-owned own work dedicated to CC0-1.0",
    expected_size=2_449_847,
    expected_sha1="ccd8f63e7250c8b7383308ec2f5bfb872f0f88df",
    width=3447,
    height=1677,
    identifiable_people=False,
    third_party_caveat="no identifiable person; exact-media review remains before training",
)

_DAGGER = _Source(
    name="cimmerian-dagger",
    source_class="knife",
    page=(
        "https://commons.wikimedia.org/w/index.php?"
        "title=File:Cimmerian_dagger.jpg&oldid=1125618414"
    ),
    url="https://upload.wikimedia.org/wikipedia/commons/8/8e/Cimmerian_dagger.jpg",
    license="CC0-1.0",
    rights_basis="uploader-owned own work dedicated to CC0-1.0",
    expected_size=2_105_888,
    expected_sha1="62c25585a1f33b734229f40c7c479d84123592b6",
    width=1159,
    height=4695,
    identifiable_people=False,
    third_party_caveat=(
        "museum-object photograph of ancient object; exact-media review remains before training"
    ),
)

_SOURCES = (_WALTHER, _SWORD, _DAGGER)


def _stream_identity(source: _Source, opener: Callable = urlopen) -> tuple[int, str, str]:
    expected = urlparse(source.url)
    if expected.scheme != "https" or expected.netloc != "upload.wikimedia.org":
        raise RuntimeError("unapproved Weapons training source URL")

    request = Request(source.url, headers={"User-Agent": _USER_AGENT})
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    total = 0

    with opener(request, timeout=20) as response:
        final = urlparse(response.geturl())
        if (
            final.scheme != "https"
            or final.netloc != expected.netloc
            or final.path != expected.path
        ):
            raise RuntimeError("Weapons training source redirected outside reviewed asset")

        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                advertised = int(content_length)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("invalid Weapons training source Content-Length") from exc
            if advertised != source.expected_size:
                raise RuntimeError("Weapons training source Content-Length changed")

        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray)):
                raise ValueError("Weapons training source response must yield bytes")
            total += len(chunk)
            if total > source.max_bytes:
                raise RuntimeError("Weapons training source exceeds pinned byte bound")
            sha1.update(chunk)
            sha256.update(chunk)

    if total != source.expected_size:
        raise RuntimeError("Weapons training source byte length changed")

    observed_sha1 = sha1.hexdigest()
    observed_sha256 = sha256.hexdigest()
    if observed_sha1 != source.expected_sha1:
        raise RuntimeError("Weapons training source SHA-1 changed")
    if source.sha256 is not None and observed_sha256 != source.sha256:
        raise RuntimeError("Weapons training source SHA-256 changed")
    return total, observed_sha1, observed_sha256


def run(opener: Callable = urlopen) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for source in _SOURCES:
        size, sha1, sha256 = _stream_identity(source, opener)
        rows.append(
            {
                "name": source.name,
                "source_class": source.source_class,
                "source_page": source.page,
                "source_url": source.url,
                "source_license": source.license,
                "rights_basis": source.rights_basis,
                "source_size": size,
                "source_sha1": sha1,
                "source_sha256": sha256,
                "sha256_pinned": source.sha256 is not None,
                "published_dimensions": [source.width, source.height],
                "identifiable_people": source.identifiable_people,
                "third_party_caveat": source.third_party_caveat,
                "decoded": False,
                "annotation_authored": False,
                "training_split_admitted": False,
            }
        )

    return {
        "evidence": "weapons-training-cc0-source-admission-v1",
        "source_count": len(rows),
        "sources": rows,
        "media_retained": False,
        "model_runtime_used": False,
        "training_run": False,
        "inference_run": False,
        "claim": (
            "source/right identity admission only; not final training authorization, "
            "annotation truth, weapon-detection accuracy, or commercial readiness"
        ),
    }


def main() -> int:
    print(json.dumps(run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
