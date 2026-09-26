"""Bounded source admission for the RF-DETR Weapons architecture fallback.

This evidence lane streams only a small exact set of source files from
roboflow/rf-detr at one pinned commit. It verifies Git blob identities and the
specific source semantics needed for a later weightless/offline RFDETRNano
construction proof. It installs no package, downloads no weights or media,
constructs no model, and performs no training or inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

REPOSITORY = "roboflow/rf-detr"
REVISION = "1985300a0e8f905c70bb1772855dcf6c964e5ed5"
LICENSE_ID = "Apache-2.0"
_ALLOWED_HOST = "raw.githubusercontent.com"
_USER_AGENT = "Analytics-lab bounded RF-DETR source admission/1.0"
_CHUNK = 64 * 1024


@dataclass(frozen=True)
class _PinnedSource:
    path: str
    size: int
    git_blob_sha1: str

    @property
    def url(self) -> str:
        return (
            f"https://{_ALLOWED_HOST}/roboflow/rf-detr/{REVISION}/"
            + quote(self.path, safe="/")
        )


_FILES = (
    _PinnedSource("LICENSE", 11_345, "56db37c53b61c7976042682a83d85383a6bf6007"),
    _PinnedSource("pyproject.toml", 27_636, "dcd04cd862d06e088f6099001925eb39cabbfd59"),
    _PinnedSource(
        "src/rfdetr/config.py",
        86_219,
        "d8fc6fa0b0ab9e96f1370cf04d6d1666f297cdb9",
    ),
    _PinnedSource(
        "src/rfdetr/detr.py",
        186_353,
        "1bea7f6092f429e1b6e07f9ccafe9a05d6ac0c19",
    ),
    _PinnedSource(
        "src/rfdetr/models/backbone/dinov2.py",
        11_904,
        "0286153cc049559e7239803c4714734a0744ba9a",
    ),
    _PinnedSource(
        "src/rfdetr/models/lwdetr.py",
        48_890,
        "69ae42d934449d552fc92ede495f7d8347a74ee6",
    ),
)


def _git_blob_sha1(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise ValueError("payload must be bytes")
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _download(source: _PinnedSource, opener: Callable = urlopen) -> bytes:
    request = Request(source.url, headers={"User-Agent": _USER_AGENT})
    expected = urlparse(source.url)
    payload = bytearray()
    with opener(request, timeout=20) as response:
        final = urlparse(response.geturl())
        if (
            final.scheme != "https"
            or final.netloc != expected.netloc
            or final.path != expected.path
        ):
            raise RuntimeError("RF-DETR source redirected outside the pinned asset")
        declared = response.headers.get("Content-Length")
        if declared is not None:
            try:
                advertised = int(declared)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("invalid RF-DETR source Content-Length") from exc
            if advertised != source.size:
                raise RuntimeError("RF-DETR source Content-Length changed")
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray)):
                raise ValueError("RF-DETR source response must yield bytes")
            payload.extend(chunk)
            if len(payload) > source.size:
                raise RuntimeError("RF-DETR source exceeded pinned size")
    data = bytes(payload)
    if len(data) != source.size:
        raise RuntimeError("RF-DETR source byte length changed")
    if _git_blob_sha1(data) != source.git_blob_sha1:
        raise RuntimeError("RF-DETR source Git blob identity changed")
    return data


def _require_static_semantics(files: dict[str, bytes]) -> None:
    try:
        license_text = files["LICENSE"].decode("utf-8")
        pyproject = files["pyproject.toml"].decode("utf-8")
        config = files["src/rfdetr/config.py"].decode("utf-8")
        detr = files["src/rfdetr/detr.py"].decode("utf-8")
        dino = files["src/rfdetr/models/backbone/dinov2.py"].decode("utf-8")
        lwdetr = files["src/rfdetr/models/lwdetr.py"].decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise RuntimeError("RF-DETR source set is incomplete or not UTF-8") from exc

    required = (
        ("LICENSE", license_text, "Apache License"),
        ("pyproject license", pyproject, 'license = {text = "Apache License 2.0"}'),
        ("pyproject version", pyproject, 'version = "1.11.0"'),
        ("nano config", config, "class RFDETRNanoConfig(RFDETRBaseConfig):"),
        ("nano patch size", config, "patch_size: int = 16"),
        ("nano resolution", config, "resolution: int = 384"),
        ("nano positional encoding", config, "positional_encoding_size: int = 24"),
        ("nano default weight name", config, 'pretrain_weights: PathLikeStr | None = "rf-detr-nano.pth"'),
        (
            "weightless checkpoint early return",
            detr,
            "if self.model_config.pretrain_weights is None:\n            return",
        ),
        (
            "backbone weight intent",
            lwdetr,
            "load_dinov2_weights=args.pretrain_weights is None,",
        ),
        ("nonstandard patch guard", dino, "if patch_size != 14:"),
        ("nonstandard patch disables DINOv2 weights", dino, "load_dinov2_weights = False"),
        (
            "local backbone constructor",
            dino,
            "else WindowedDinov2WithRegistersBackbone(windowed_dino_config)",
        ),
    )
    for label, text, needle in required:
        if needle not in text:
            raise RuntimeError(f"RF-DETR pinned source semantics changed: {label}")


def run(opener: Callable = urlopen) -> dict[str, object]:
    files: dict[str, bytes] = {}
    identities: list[dict[str, object]] = []
    for source in _FILES:
        payload = _download(source, opener)
        files[source.path] = payload
        identities.append(
            {
                "path": source.path,
                "size": len(payload),
                "git_blob_sha1": _git_blob_sha1(payload),
            }
        )
    _require_static_semantics(files)
    return {
        "evidence": "weapons-rfdetr-source-admission-v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": LICENSE_ID,
        "files": identities,
        "file_count": len(identities),
        "pretrained_weights_admitted": False,
        "model_constructed": False,
        "training_run": False,
        "inference_run": False,
        "network_during_future_model_construction_authorized": False,
        "claim": (
            "source/license and weightless-construction-path admission only; "
            "no RF-DETR weights, model execution, training, accuracy or readiness claim"
        ),
    }


def main() -> int:
    print(json.dumps(run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
