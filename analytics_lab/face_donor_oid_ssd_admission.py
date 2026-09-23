"""Bounded discovery/admission probe for Google's standard OIDv4 SSD MobileNetV2.

The first evidence head is discovery-only: it may download exactly one official
Google TensorFlow model-zoo archive into memory, bound its size, compute hashes,
and inspect tar member metadata/hashes without extraction, inference, or media.
It MUST NOT report the artifact as admitted until exact archive and member
identities are pinned in this module on a later head.
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
    "ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz"
)
_ALLOWED_HOST = "storage.googleapis.com"
_MAX_ARCHIVE_BYTES = 250_000_000
_USER_AGENT = "Analytics-Lab-face-oid-ssd-admission/1"

# Provenance is pinned independently of the archive-discovery result. This
# TensorFlow Models revision contains the corrected OIDv4 label map and the
# official model-zoo entry for the 2018-12-12 archive.
_TF_MODELS_REVISION = "0558408514dacf2fe2860cd72ac56cbdf62a24c0"
_TF_MODELS_LICENSE = "Apache-2.0"
_TF_MODELS_LICENSE_BLOB = "43fcf7bf1f1f9f824a1debf05d6ced45bf5810aa"
_OID_V4_LABEL_MAP_PATH = "research/object_detection/data/oid_v4_label_map.pbtxt"
_OID_V4_LABEL_MAP_BLOB = "643b9e8ed5d9239a3248b895fb32f3b51caa92f3"
_HUMAN_FACE_CLASS_ID = 502
_HUMAN_FACE_MID = "/m/0dzct"
_HUMAN_FACE_DISPLAY_NAME = "Human face"
_OPEN_IMAGES_V4_LICENSE_URL = "https://storage.googleapis.com/openimages/web/factsfigures_v4.html"
_OPEN_IMAGES_LICENSE_CAVEAT = (
    "annotations CC BY 4.0; images listed CC BY 2.0; Google makes no "
    "representations or warranties regarding each image's license status and "
    "requires users to verify each image license themselves"
)

# Discovery head intentionally leaves these unset. The first *admission* head
# must replace them with the exact evidence observed from the official archive.
_EXPECTED_ARCHIVE_SIZE: int | None = None
_EXPECTED_ARCHIVE_SHA256: str | None = None
_EXPECTED_ARCHIVE_MEMBERS: tuple[tuple[str, int, str], ...] | None = None


def _declared_size() -> int | None:
    parsed = urllib.parse.urlparse(_ARCHIVE_URL)
    if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST:
        raise RuntimeError("unapproved OID SSD donor URL")
    request = urllib.request.Request(
        _ARCHIVE_URL,
        headers={"User-Agent": _USER_AGENT},
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        final = urllib.parse.urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != _ALLOWED_HOST:
            raise RuntimeError("OID SSD donor redirect escaped approved host")
        declared = response.headers.get("Content-Length")
    if declared is None:
        return None
    try:
        value = int(declared)
    except ValueError as exc:
        raise RuntimeError("invalid Content-Length") from exc
    if value < 1 or value > _MAX_ARCHIVE_BYTES:
        raise RuntimeError("OID SSD donor declared size outside bound")
    return value


def _download() -> bytes:
    parsed = urllib.parse.urlparse(_ARCHIVE_URL)
    if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST:
        raise RuntimeError("unapproved OID SSD donor URL")
    request = urllib.request.Request(_ARCHIVE_URL, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        final = urllib.parse.urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != _ALLOWED_HOST:
            raise RuntimeError("OID SSD donor redirect escaped approved host")
        declared = response.headers.get("Content-Length")
        if declared is not None:
            try:
                declared_size = int(declared)
            except ValueError as exc:
                raise RuntimeError("invalid Content-Length") from exc
            if declared_size < 1 or declared_size > _MAX_ARCHIVE_BYTES:
                raise RuntimeError("OID SSD donor declared size outside bound")
        payload = response.read(_MAX_ARCHIVE_BYTES + 1)
    if not payload or len(payload) > _MAX_ARCHIVE_BYTES:
        raise RuntimeError("OID SSD donor archive outside size bound")
    return payload


def _safe_members(payload: bytes) -> list[dict[str, object]]:
    members: list[dict[str, object]] = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if len(members) >= 128:
                raise RuntimeError("OID SSD donor archive member count exceeds bound")
            name = member.name
            if (
                not isinstance(name, str)
                or not name
                or name.startswith("/")
                or "\\" in name
                or ".." in name.split("/")
            ):
                raise RuntimeError("unsafe OID SSD donor archive member path")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError("unsafe OID SSD donor archive member type")
            if member.isfile() and member.size > _MAX_ARCHIVE_BYTES:
                raise RuntimeError("OID SSD donor archive member exceeds size bound")

            item: dict[str, object] = {
                "name": name,
                "size": int(member.size),
                "type": "file" if member.isfile() else "dir" if member.isdir() else "other",
            }
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError("OID SSD donor archive member unreadable")
                digest = hashlib.sha256()
                seen = 0
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    seen += len(chunk)
                    if seen > member.size:
                        raise RuntimeError("OID SSD donor archive member length mismatch")
                    digest.update(chunk)
                if seen != member.size:
                    raise RuntimeError("OID SSD donor archive member length mismatch")
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
            raise RuntimeError("OID SSD donor archive member hash missing")
        identity.append((str(item["name"]), int(item["size"]), digest))
    return tuple(identity)


def _relevant_members(members: list[dict[str, object]]) -> list[dict[str, object]]:
    suffixes = (
        "frozen_inference_graph.pb",
        "pipeline.config",
        "graph.pbtxt",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "label_map.pbtxt",
    )
    return [
        item for item in members
        if item["type"] == "file" and str(item["name"]).endswith(suffixes)
    ]


def _provenance() -> dict[str, object]:
    return {
        "tensorflow_models_revision": _TF_MODELS_REVISION,
        "tensorflow_models_license": _TF_MODELS_LICENSE,
        "tensorflow_models_license_blob": _TF_MODELS_LICENSE_BLOB,
        "oid_v4_label_map_path": _OID_V4_LABEL_MAP_PATH,
        "oid_v4_label_map_blob": _OID_V4_LABEL_MAP_BLOB,
        "human_face_class": {
            "id": _HUMAN_FACE_CLASS_ID,
            "mid": _HUMAN_FACE_MID,
            "display_name": _HUMAN_FACE_DISPLAY_NAME,
        },
        "open_images_v4_license_url": _OPEN_IMAGES_V4_LICENSE_URL,
        "open_images_license_caveat": _OPEN_IMAGES_LICENSE_CAVEAT,
    }


def run() -> dict[str, object]:
    declared = _declared_size()
    payload = _download()
    if declared is not None and declared != len(payload):
        raise RuntimeError("OID SSD donor archive size differs from HEAD declaration")

    sha256 = hashlib.sha256(payload).hexdigest()
    members = _safe_members(payload)
    identity = _member_identity(members)
    relevant = _relevant_members(members)
    if not any(str(item["name"]).endswith("frozen_inference_graph.pb") for item in relevant):
        raise RuntimeError("OID SSD donor frozen graph missing")
    if not any(str(item["name"]).endswith("pipeline.config") for item in relevant):
        raise RuntimeError("OID SSD donor pipeline config missing")

    pinned = (
        _EXPECTED_ARCHIVE_SIZE is not None
        and _EXPECTED_ARCHIVE_SHA256 is not None
        and _EXPECTED_ARCHIVE_MEMBERS is not None
    )
    if pinned:
        if len(payload) != _EXPECTED_ARCHIVE_SIZE:
            raise RuntimeError("OID SSD donor archive size mismatch")
        if sha256 != _EXPECTED_ARCHIVE_SHA256:
            raise RuntimeError("OID SSD donor archive SHA-256 mismatch")
        if identity != _EXPECTED_ARCHIVE_MEMBERS:
            raise RuntimeError("OID SSD donor archive member identity mismatch")

    result = {
        "evidence": "face-donor-oid-ssd-admission-v1",
        "source_url": _ARCHIVE_URL,
        "archive_bytes": len(payload),
        "archive_sha256": sha256,
        "archive_pinned": pinned,
        "members_pinned": pinned,
        "admitted": pinned,
        "members": members,
        "relevant_members": relevant,
        "total_members": len(members),
        "provenance": _provenance(),
        "inference_run": False,
        "media_used": False,
        "artifact_retained": False,
        "claim": (
            "immutable artifact/provenance admission only; not face-detection accuracy"
            if pinned
            else "discovery only; NOT admitted and not face-detection accuracy"
        ),
    }
    return result


def main() -> int:
    print(json.dumps(run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
