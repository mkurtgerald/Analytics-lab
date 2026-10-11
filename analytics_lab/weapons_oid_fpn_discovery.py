"""Metadata-only streaming discovery of one unpinned official OIDv4 SSD FPN archive.

No native dependency, model construction, inference, extraction or file retention
occurs. Only pipeline.config (at most 64 KiB) is buffered; every other member is
hashed in chunks. Fixed tar headers reject extended/sparse/link metadata before
payload allocation. Discovery identities require a later independent pinning gate.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import PurePosixPath
import re
import tarfile
from time import monotonic
from typing import Callable
from urllib.request import HTTPRedirectHandler, Request, build_opener

from . import face_donor_oid_ssd_admission as provenance

_ROOT = "ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20"
_WRAPPER_PREFIX = "ssd_resnet101_v1_fpn_shared_box_predictor"
_ARCHIVE_URL = "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/" + _ROOT + ".tar.gz"
_EXPECTED_SIZE = 638_212_759  # Prior HEAD observation, not a content-identity admission.
_MAX_COMPRESSED = 650_000_000
_MAX_EXPANDED = 2_000_000_000
_MAX_REGULAR = 700_000_000
_MAX_MEMBERS = 128
_MAX_CONFIG = 64 * 1024
_CHUNK = 64 * 1024
_TIME_LIMIT = 210
_READ_TIMEOUT = 30


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("FPN discovery redirects are prohibited")


def _remaining(deadline: float, clock: Callable) -> float:
    value = deadline - clock()
    if value <= 0:
        raise TimeoutError("FPN discovery deadline exceeded")
    return value


class _Compressed:
    def __init__(self, response, deadline: float, clock: Callable):
        self.response, self.deadline, self.clock = response, deadline, clock
        setter = getattr(response, "settimeout", None)
        if not callable(setter):
            socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
            setter = getattr(socket, "settimeout", None)
        if not callable(setter):
            raise RuntimeError("response does not support bounded per-read timeouts")
        self.reader = getattr(response, "read1", None)
        if not callable(self.reader):
            raise RuntimeError("response must expose one-receive read1")
        self.setter = setter
        self.count, self.digest, self.eof = 0, hashlib.sha256(), False

    def read(self, size: int = -1) -> bytes:
        _remaining(self.deadline, self.clock)
        if self.eof or size == 0:
            return b""
        requested = min(_CHUNK, size if size > 0 else _CHUNK, _MAX_COMPRESSED - self.count + 1)
        collected = bytearray()
        while len(collected) < requested and not self.eof:
            self.setter(min(_READ_TIMEOUT, _remaining(self.deadline, self.clock)))
            data = self.reader(requested - len(collected))
            _remaining(self.deadline, self.clock)
            if type(data) is not bytes or len(data) > requested - len(collected):
                raise RuntimeError("malformed bounded HTTP read")
            self.count += len(data)
            if self.count > _MAX_COMPRESSED:
                raise RuntimeError("compressed archive exceeds its byte ceiling")
            self.digest.update(data)
            collected.extend(data)
            self.eof = not data or bool(getattr(self.response, "isclosed", lambda: False)())
        return bytes(collected)


class _Expanded:
    def __init__(self, stream, deadline: float, clock: Callable):
        self.stream, self.deadline, self.clock, self.count = stream, deadline, clock, 0

    def read(self, size: int) -> bytes:
        _remaining(self.deadline, self.clock)
        data = self.stream.read(min(size, _CHUNK, _MAX_EXPANDED - self.count + 1))
        _remaining(self.deadline, self.clock)
        self.count += len(data)
        if self.count > _MAX_EXPANDED:
            raise RuntimeError("expanded archive exceeds its byte ceiling")
        return data

    def exact(self, size: int) -> bytes:
        # Used only for 512-byte headers and <=511-byte padding.
        parts = []
        remaining = size
        while remaining:
            data = self.read(remaining)
            if not data:
                raise RuntimeError("truncated tar member/header")
            parts.append(data)
            remaining -= len(data)
        return b"".join(parts)


def _safe_header(block: bytes):
    member = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
    original = block[:100].split(b"\0", 1)[0].decode("utf-8", "strict")
    prefix = block[345:500].split(b"\0", 1)[0].decode("utf-8", "strict")
    if prefix:
        original = prefix + "/" + original
    name = original[2:] if original.startswith("./") else original
    if member.type == tarfile.DIRTYPE and name.endswith("/"):
        name = name[:-1]
    if member.type == tarfile.DIRTYPE and original in (".", "./"):
        name = "."
    path = PurePosixPath(name)
    if (not name or len(name) > 256 or name != path.as_posix() or path.is_absolute()
            or "\\" in name or any(ord(ch) < 32 or ord(ch) == 127 for ch in name)
            or ".." in path.parts or name == "." and member.type != tarfile.DIRTYPE):
        raise RuntimeError("unsafe or noncanonical relative tar path")
    if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE) or member.linkname:
        raise RuntimeError("unsupported extended/link/sparse/special tar header")
    if (type(member.size) is not int or member.size < 0 or member.size > _MAX_REGULAR
            or member.type == tarfile.DIRTYPE and member.size != 0):
        raise RuntimeError("tar member size/type exceeds its boundary")
    member.name = name
    return member, original


def _block(text: str, name: str) -> str:
    matches = tuple(re.finditer(r"\b" + re.escape(name) + r"\s*\{", text))
    if len(matches) != 1:
        raise RuntimeError("pipeline block missing or ambiguous: " + name)
    start, depth = matches[0].end(), 1
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index]
    raise RuntimeError("unbalanced pipeline config block")


def _number(text: str, field: str) -> int:
    values = re.findall(r"\b" + re.escape(field) + r"\s*:\s*([0-9]{1,6})(?=\s|[,;}]|$)", text)
    if len(values) != 1 or not 1 <= int(values[0]) <= 32768:
        raise RuntimeError("pipeline integer missing, ambiguous or outside bound: " + field)
    return int(values[0])


def _pipeline(data: bytes) -> dict:
    text = data.decode("utf-8", "strict")
    # Remove quoted strings/comments before inspecting the small text-proto surface.
    text = re.sub(r'"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27|#[^\n]*', " ", text)
    ssd = _block(_block(text, "model"), "ssd")
    resized = _block(ssd, "image_resizer")
    kinds = [name for name in ("fixed_shape_resizer", "keep_aspect_ratio_resizer")
             if re.search(r"\b" + name + r"\s*\{", resized)]
    if len(kinds) != 1:
        raise RuntimeError("pipeline resizer missing or ambiguous")
    kind = kinds[0]
    body = _block(resized, kind)
    fields = ("height", "width") if kind == "fixed_shape_resizer" else ("min_dimension", "max_dimension")
    resizer = {"type": kind, **{field: _number(body, field) for field in fields}}
    if kind == "keep_aspect_ratio_resizer":
        values = re.findall(r"\bpad_to_max_dimension\s*:\s*(true|false)\b", body)
        if len(values) > 1:
            raise RuntimeError("pipeline padding field is ambiguous")
        resizer["pad_to_max_dimension"] = values[0] == "true" if values else None
    return {"num_classes": _number(ssd, "num_classes"), "image_resizer": resizer}


def _members(expanded: _Expanded) -> tuple[list, bytes, str, dict]:
    members, seen, config = [], set(), None
    while True:
        block = expanded.exact(512)
        if block == bytes(512):
            if expanded.exact(512) != bytes(512):
                raise RuntimeError("tar requires two zero end blocks")
            while True:
                padding = expanded.read(_CHUNK)
                if not padding:
                    break
                if any(padding):
                    raise RuntimeError("nonzero expanded data follows tar end")
            break
        if len(members) == _MAX_MEMBERS:
            raise RuntimeError("tar member count exceeds its ceiling")
        member, original = _safe_header(block)
        if member.name in seen:
            raise RuntimeError("duplicate tar path")
        seen.add(member.name)
        record = {"name": member.name, "original_name": original,
                  "type": "directory" if member.isdir() else "file", "size": member.size}
        if not member.isdir():
            is_config = PurePosixPath(member.name).name == "pipeline.config"
            if is_config and config is not None:
                raise RuntimeError("pipeline config basename is ambiguous")
            if is_config and member.size > _MAX_CONFIG:
                raise RuntimeError("pipeline config exceeds its capture ceiling")
            digest, remaining, captured = hashlib.sha256(), member.size, bytearray()
            while remaining:
                data = expanded.read(min(remaining, _CHUNK))
                if not data:
                    raise RuntimeError("truncated regular tar member")
                digest.update(data)
                if is_config:
                    captured.extend(data)
                remaining -= len(data)
            record["sha256"] = digest.hexdigest()
            if is_config:
                config = bytes(captured)
            if any(expanded.exact((-member.size) % 512)):
                raise RuntimeError("nonzero regular-member padding")
        members.append(record)
    required = []
    for basename in ("frozen_inference_graph.pb", "pipeline.config"):
        files = [m for m in members if m["type"] == "file" and PurePosixPath(m["name"]).name == basename]
        if len(files) != 1 or files[0]["size"] <= 0:
            raise RuntimeError("required graph/config basename is missing, empty or ambiguous")
        required.append(files[0])
    root = str(PurePosixPath(required[0]["name"]).parent)
    if (str(PurePosixPath(required[1]["name"]).parent) != root
            or root != "." and (len(PurePosixPath(root).parts) != 1 or not root.startswith(_WRAPPER_PREFIX))):
        raise RuntimeError("graph/config must share the flat or one approved wrapping root")
    if root != "." and any(m["name"] != "." and not (m["name"].startswith(root + "/")
                                                      or m["name"] == root and m["type"] == "directory") for m in members):
        raise RuntimeError("archive member lies outside its discovered wrapping root")
    return members, config, root, required[0]


def run(*, opener: Callable | None = None, clock: Callable = monotonic) -> dict:
    deadline = clock() + _TIME_LIMIT
    open_url = opener or build_opener(_NoRedirect()).open
    request = Request(_ARCHIVE_URL, headers={"User-Agent": "Analytics-lab-weapons-fpn-discovery/1",
                                           "Accept-Encoding": "identity"})
    with open_url(request, timeout=min(_READ_TIMEOUT, _remaining(deadline, clock))) as response:
        if response.geturl() != _ARCHIVE_URL or getattr(response, "status", None) != 200:
            raise RuntimeError("archive response escaped the exact approved URL/status")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise RuntimeError("HTTP content encoding must preserve archive bytes")
        declared = response.headers.get("Content-Length")
        if declared is not None and (not re.fullmatch(r"[0-9]{1,15}", declared)
                                     or int(declared) != _EXPECTED_SIZE or int(declared) > _MAX_COMPRESSED):
            raise RuntimeError("archive declared size differs from the observed HEAD/bound")
        compressed = _Compressed(response, deadline, clock)
        try:
            with gzip.GzipFile(fileobj=compressed, mode="rb") as stream:
                expanded = _Expanded(stream, deadline, clock)
                members, config, root, graph = _members(expanded)
            while compressed.read(_CHUNK):
                pass
        except TimeoutError:
            raise
        except (OSError, EOFError, tarfile.HeaderError, UnicodeError) as exc:
            raise RuntimeError("invalid/truncated archive or UTF-8 metadata") from exc
        if compressed.count != _EXPECTED_SIZE:
            raise RuntimeError("complete compressed archive size differs from the observed HEAD")
        try:
            pipeline = _pipeline(config)
        except UnicodeError as exc:
            raise RuntimeError("pipeline config is not UTF-8") from exc
        _remaining(deadline, clock)
        return {"evidence": "weapons-oid-fpn-discovery-v1", "archive_url": _ARCHIVE_URL,
                "archive_size": compressed.count, "archive_sha256": compressed.digest.hexdigest(),
                "expanded_size": expanded.count, "members": members, "member_count": len(members),
                "archive_root": root, "graph": graph,
                "pipeline": pipeline, "archive_identity_pinned": False, "artifact_admitted": False,
                "tensorflow_models_revision": provenance._TF_MODELS_REVISION,
                "tensorflow_models_license": provenance._TF_MODELS_LICENSE,
                "tensorflow_models_license_blob": provenance._TF_MODELS_LICENSE_BLOB,
                "oid_v4_label_map_path": provenance._OID_V4_LABEL_MAP_PATH,
                "oid_v4_label_map_blob": provenance._OID_V4_LABEL_MAP_BLOB,
                "open_images_license_caveat": provenance._OPEN_IMAGES_LICENSE_CAVEAT,
                "native_installed": False, "model_constructed": False, "inference_run": False,
                "artifact_retained": False, "accuracy_claim": False, "readiness_claim": False}


def main() -> int:
    print(json.dumps(run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
