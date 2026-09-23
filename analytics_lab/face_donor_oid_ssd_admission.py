"""Bounded admission probe for Google's standard OIDv4 SSD MobileNetV2.

The official archive is downloaded only in memory, fail-closed against exact
archive and regular-member identities, and discarded. This module performs no
network construction, inference, media processing, extraction, or retention.
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

# Exact hosted discovery evidence from Analytics quality run #259.
_EXPECTED_ARCHIVE_SIZE = 158_851_107
_EXPECTED_ARCHIVE_SHA256 = "8dd82cc52625eb9b43c6ed8050868fe1e78f878c6f028519fc731b3c7e9035f7"
_EXPECTED_ARCHIVE_MEMBERS = (
    (
        "ssd_mobilenet_v2_oid_v4_2018_12_12/model.ckpt.meta",
        11_915_505,
        "541aebeef5c674609f4b2cc229c265b053dc7df3226143b01037c0a42d715f7f",
    ),
    (
        "ssd_mobilenet_v2_oid_v4_2018_12_12/checkpoint",
        77,
        "dd1b025d2e155283f5e300ce95bf6d5b6bc0f7fe010db73daa6975eb896ab9cb",
    ),
    (
        "ssd_mobilenet_v2_oid_v4_2018_12_12/frozen_inference_graph.pb",
        66_606_111,
        "150f3eb77b741ed24e1a19559589205ccd616b059fc976c8b8d5cefc126bf86b",
    ),
    (
        "ssd_mobilenet_v2_oid_v4_2018_12_12/saved_model/saved_model.pb",
        67_889_548,
        "f5453b9c2bb73be4d21eef9eb37a8fce0a2d09903ea1aecd44ff49fa6efe258f",
    ),
    (
        "ssd_mobilenet_v2_oid_v4_2018_12_12/model.ckpt.index",
        14_175,
        "0557da0b4b7d555fc1483d3ba3a89732c7606e7176c30839335333f48b9d81f4",
    ),
    (
        "ssd_mobilenet_v2_oid_v4_2018_12_12/pipeline.config",
        4_267,
        "cf424b06dabcc7acd6bf71ffd941c5a9890b975f8036444e03a2290391a24614",
    ),
    (
        "ssd_mobilenet_v2_oid_v4_2018_12_12/model.ckpt.data-00000-of-00001",
        57_841_536,
        "61bc5931d1cc44cd83b80ebee3cb75757e84c1a49ac5cea71149e36a93da4044",
    ),
)

# Provenance is pinned independently of archive identity. This TensorFlow
# Models revision contains the official model-zoo entry and corrected OIDv4
# label map. The archive itself contains no label-map member.
_TF_MODELS_REVISION = "0558408514dacf2fe2860cd72ac56cbdf62a24c0"
_TF_MODELS_LICENSE = "Apache-2.0"
_TF_MODELS_LICENSE_BLOB = "43fcf7bf1f1f9f824a1debf05d6ced45bf5810aa"
_OID_V4_LABEL_MAP_PATH = "research/object_detection/data/oid_v4_label_map.pbtxt"
_OID_V4_LABEL_MAP_BLOB = "643b9e8ed5d9239a3248b895fb32f3b51caa92f3"
_ARCHIVE_LABEL_MAP_MEMBER = None
_HUMAN_FACE_CLASS_ID = 502
_HUMAN_FACE_MID = "/m/0dzct"
_HUMAN_FACE_DISPLAY_NAME = "Human face"
_OPEN_IMAGES_V4_LICENSE_URL = "https://storage.googleapis.com/openimages/web/factsfigures_v4.html"
_OPEN_IMAGES_LICENSE_CAVEAT = (
    "annotations CC BY 4.0; images listed CC BY 2.0; Google makes no "
    "representations or warranties regarding each image's license status and "
    "requires users to verify each image license themselves"
)


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
        "saved_model.pb",
        "pipeline.config",
        "checkpoint",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "label_map.pbtxt",
    )
    return [
        item
        for item in members
        if item["type"] == "file" and str(item["name"]).endswith(suffixes)
    ]


def _provenance() -> dict[str, object]:
    return {
        "tensorflow_models_revision": _TF_MODELS_REVISION,
        "tensorflow_models_license": _TF_MODELS_LICENSE,
        "tensorflow_models_license_blob": _TF_MODELS_LICENSE_BLOB,
        "oid_v4_label_map_path": _OID_V4_LABEL_MAP_PATH,
        "oid_v4_label_map_blob": _OID_V4_LABEL_MAP_BLOB,
        "archive_label_map_member": _ARCHIVE_LABEL_MAP_MEMBER,
        "label_map_source": "external_corrected_tensorflow_models_revision",
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
    if declared is not None and declared != _EXPECTED_ARCHIVE_SIZE:
        raise RuntimeError("OID SSD donor archive size mismatch")

    payload = _download()
    if len(payload) != _EXPECTED_ARCHIVE_SIZE:
        raise RuntimeError("OID SSD donor archive size mismatch")
    sha256 = hashlib.sha256(payload).hexdigest()
    if sha256 != _EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("OID SSD donor archive SHA-256 mismatch")

    members = _safe_members(payload)
    identity = _member_identity(members)
    if identity != _EXPECTED_ARCHIVE_MEMBERS:
        raise RuntimeError("OID SSD donor archive member identity mismatch")

    archive_label_maps = [
        str(item["name"])
        for item in members
        if item["type"] == "file" and str(item["name"]).endswith("label_map.pbtxt")
    ]
    if archive_label_maps:
        raise RuntimeError("unexpected OID SSD archive label-map member")

    relevant = _relevant_members(members)
    if not any(str(item["name"]).endswith("frozen_inference_graph.pb") for item in relevant):
        raise RuntimeError("OID SSD donor frozen graph missing")
    if not any(str(item["name"]).endswith("pipeline.config") for item in relevant):
        raise RuntimeError("OID SSD donor pipeline config missing")
    if not any(str(item["name"]).endswith("saved_model.pb") for item in relevant):
        raise RuntimeError("OID SSD donor SavedModel graph missing")

    return {
        "evidence": "face-donor-oid-ssd-admission-v2",
        "source_url": _ARCHIVE_URL,
        "archive_bytes": len(payload),
        "archive_sha256": sha256,
        "archive_pinned": True,
        "members_pinned": True,
        "admitted": True,
        "members": members,
        "relevant_members": relevant,
        "total_members": len(members),
        "provenance": _provenance(),
        "inference_run": False,
        "media_used": False,
        "artifact_retained": False,
        "claim": "immutable artifact/provenance admission only; not face-detection accuracy",
    }


def main() -> int:
    print(json.dumps(run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
