"""Bounded frame-identity evidence for the pre-registered CC0 tracking window.

This module is intentionally narrower than a benchmark runner. On the dedicated
``evidence/tracking-cc0-benchmark-*`` lane it downloads only the already
rights-reviewed Wikimedia source into the hosted runner's ephemeral temp
folder, verifies the exact admitted bytes, decodes sequentially only through the
end of the fixed 25-frame window, and emits cryptographic frame identities.

It never uploads or commits media, runs a detector/tracker, authors ground
truth, tunes parameters, or reports tracking accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import BinaryIO, Callable, Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .tracking_benchmark_plan import TrackingBenchmarkPlan, wikimedia_pedestrian_smoke_plan
from .tracking_evidence import FrameDigest, canonical_frame_manifest_sha256
from .tracking_ground_truth_package import canonical_rgb24_frame_sha256
from .wikimedia_tracking_admission import (
    AUTHOR,
    LICENSE_EXPRESSION,
    MAX_BYTES,
    PUBLISHED_SHA1,
    PUBLISHED_SHA256,
    PUBLISHED_SIZE,
    RIGHTS_SOURCE,
    SOURCE_PAGE,
    SOURCE_URL,
)

CHUNK_BYTES = 64 * 1024
BENCHMARK_BRANCH_PREFIX = "evidence/tracking-cc0-benchmark-"
PINNED_OPENCV_VERSION = "4.12.0"


@dataclass(frozen=True)
class VerifiedSource:
    path: Path
    size_bytes: int
    sha1: str
    sha256: str


def _verify_final_url(final_url: str) -> None:
    parsed = urlparse(final_url)
    expected = urlparse(SOURCE_URL)
    if (
        parsed.scheme != "https"
        or parsed.netloc != expected.netloc
        or parsed.path != expected.path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("evidence source redirected outside the reviewed asset")


def download_verified_source(
    destination: Path,
    *,
    opener: Callable[..., BinaryIO] = urlopen,
    max_bytes: int = MAX_BYTES,
) -> VerifiedSource:
    """Download the exact admitted source to one caller-owned ephemeral path."""

    if not isinstance(destination, Path):
        raise ValueError("destination must be a pathlib.Path")
    if type(max_bytes) is not int or max_bytes < PUBLISHED_SIZE:
        raise ValueError("max_bytes must accommodate the exact admitted source")
    if destination.exists():
        raise RuntimeError("destination must not already exist")

    request = Request(
        SOURCE_URL,
        headers={"User-Agent": "Analytics-Lab-tracking-frame-evidence/1.0"},
        method="GET",
    )
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    total = 0
    try:
        with opener(request, timeout=30) as response:
            _verify_final_url(response.geturl())
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    advertised = int(content_length)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("invalid evidence Content-Length") from exc
                if advertised != PUBLISHED_SIZE:
                    raise RuntimeError("evidence Content-Length does not match admitted source")

            with destination.open("xb") as handle:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray)):
                        raise RuntimeError("evidence response must yield bytes")
                    total += len(chunk)
                    if total > max_bytes:
                        raise RuntimeError("evidence asset exceeded bounded download size")
                    handle.write(chunk)
                    sha1.update(chunk)
                    sha256.update(chunk)

        sha1_hex = sha1.hexdigest()
        sha256_hex = sha256.hexdigest()
        if total != PUBLISHED_SIZE:
            raise RuntimeError("evidence byte length does not match admitted source")
        if sha1_hex != PUBLISHED_SHA1:
            raise RuntimeError("evidence SHA-1 does not match admitted source")
        if sha256_hex != PUBLISHED_SHA256:
            raise RuntimeError("evidence SHA-256 does not match admitted source")
        return VerifiedSource(destination, total, sha1_hex, sha256_hex)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def hash_rgb24_sequence(
    plan: TrackingBenchmarkPlan,
    frames: Iterable[tuple[int, bytes]],
) -> tuple[tuple[FrameDigest, ...], str]:
    """Hash an exact ordered RGB24 sequence for one benchmark plan."""

    if not isinstance(plan, TrackingBenchmarkPlan):
        raise ValueError("plan must be a TrackingBenchmarkPlan")
    try:
        values = tuple(frames)
    except TypeError as exc:
        raise ValueError("frames must be iterable") from exc
    expected = plan.frame_indices
    actual = tuple(item[0] for item in values)
    if actual != expected:
        raise ValueError("RGB24 frames must match the complete benchmark window in exact order")
    digests = tuple(
        FrameDigest(index, canonical_rgb24_frame_sha256(plan, index, rgb24))
        for index, rgb24 in values
    )
    return digests, canonical_frame_manifest_sha256(digests, max_frames=plan.frame_count)


def decode_planned_frames(source: Path, plan: TrackingBenchmarkPlan) -> tuple[tuple[FrameDigest, ...], str, str]:
    """Sequentially decode through the fixed window and hash only planned frames."""

    if not isinstance(source, Path) or not source.is_file():
        raise ValueError("source must be an existing file")
    if not isinstance(plan, TrackingBenchmarkPlan):
        raise ValueError("plan must be a TrackingBenchmarkPlan")

    import cv2  # Lazy: normal regression paths do not require this evidence dependency.

    if cv2.__version__ != PINNED_OPENCV_VERSION:
        raise RuntimeError("OpenCV runtime version does not match the reviewed evidence pin")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("unable to open admitted evidence source")
    wanted = set(plan.frame_indices)
    frames: list[tuple[int, bytes]] = []
    last_index = plan.frame_indices[-1]
    index = 0
    try:
        while index <= last_index:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("decoder ended before the benchmark window completed")
            if index in wanted:
                if getattr(frame, "shape", None) != (plan.image_height, plan.image_width, 3):
                    raise RuntimeError("decoded frame dimensions do not match benchmark plan")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((index, rgb.tobytes(order="C")))
            index += 1
    finally:
        capture.release()

    digests, manifest = hash_rgb24_sequence(plan, frames)
    return digests, manifest, cv2.__version__


def _require_benchmark_branch() -> None:
    branch = os.environ.get("GITHUB_HEAD_REF", "")
    if not branch.startswith(BENCHMARK_BRANCH_PREFIX):
        raise RuntimeError("tracking frame evidence may run only on the dedicated evidence branch")


def main() -> int:
    _require_benchmark_branch()
    plan = wikimedia_pedestrian_smoke_plan()
    runner_temp = os.environ.get("RUNNER_TEMP")
    if not runner_temp:
        raise RuntimeError("RUNNER_TEMP is required for ephemeral evidence execution")
    temp_root = Path(runner_temp).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)

    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=temp_root,
            prefix="analytics-tracking-cc0-",
            suffix=".webm",
            delete=False,
        ) as handle:
            path = Path(handle.name)
        path.unlink()  # download_verified_source requires create-exclusive semantics.
        source = download_verified_source(path)
        digests, manifest, decoder_version = decode_planned_frames(source.path, plan)
        print(json.dumps({
            "source_page": SOURCE_PAGE,
            "rights_source": RIGHTS_SOURCE,
            "source_url": SOURCE_URL,
            "license": LICENSE_EXPRESSION,
            "author": AUTHOR,
            "commercial_evaluation_authorized": True,
            "source_sha256": source.sha256,
            "benchmark_plan_sha256": plan.canonical_sha256(),
            "frame_indices": [item.frame_index for item in digests],
            "frame_sha256": [item.sha256 for item in digests],
            "frame_manifest_sha256": manifest,
            "decoder": "opencv-python-headless",
            "decoder_version": decoder_version,
            "decoded_frames_persisted": False,
            "source_media_retained": False,
            "tracker_executed": False,
            "accuracy_claim": False,
        }, sort_keys=True))
        return 0
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
