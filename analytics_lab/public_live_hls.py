"""Bounded metadata/TS acquisition for one reviewed Caltrans public camera.

No decoder, inference, frame export or media retention policy lives here. The
caller owns a new RUNNER_TEMP directory and its cleanup. Only three complete,
sync-checked TS files are written after a second playlist proves advancement.
HTTP deadlines are cooperative; open/header timeouts bound inactivity.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

CATALOG_URL = "https://cwwp2.dot.ca.gov/data/d1/cctv/cctvStatusD01.json"
MASTER_URL = "https://wzmedia.dot.ca.gov/D1/eureka_14th_320x240.stream/playlist.m3u8"
CAMERA_NAME = "US-101 : Eureka / 14th - Looking North (C039)"
_CATALOG_CAP = 2_000_000
_PLAYLIST_CAP = 64 * 1024
_SEGMENT_CAP = 2 * 1024 * 1024
_TOTAL_SEGMENT_CAP = 4 * 1024 * 1024


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LiveDeadline:
    """Shared cooperative budget; callers may reserve cleanup outside it."""

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 total_seconds: float = 180.0) -> None:
        if (not callable(clock) or type(total_seconds) not in (int, float)
                or not math.isfinite(total_seconds) or not 0 < total_seconds <= 180):
            raise ValueError("deadline requires a clock and a budget in (0, 180]")
        self.clock, self.total_seconds = clock, float(total_seconds)
        self.started = clock()
        self.timeout()

    def timeout(self, cap: float = 10.0) -> float:
        elapsed = self.clock() - self.started
        if (not math.isfinite(elapsed) or elapsed < 0 or self.total_seconds - elapsed <= 0):
            raise TimeoutError("public-live cooperative deadline exhausted")
        if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
            raise ValueError("timeout cap must be finite and positive")
        return min(float(cap), self.total_seconds - elapsed)

    def check(self) -> None:
        self.timeout()


@dataclass(frozen=True)
class FetchedBytes:
    payload: bytes
    sha256: str
    size_bytes: int
    http_date: str | None
    received_utc: str


@dataclass(frozen=True)
class LiveSegment:
    sequence: int
    duration_seconds: float
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class LiveWindow:
    segments: tuple[LiveSegment, ...]
    metadata: dict[str, Any]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise RuntimeError("public-live redirects are prohibited")


def _open(request: Request, *, timeout: float):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def _safe_url(url: str) -> None:
    if type(url) is not str or len(url) > 2048 or any(ord(c) <= 32 or ord(c) >= 127 for c in url):
        raise ValueError("URL must be bounded ASCII HTTPS")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.netloc != parsed.hostname
            or parsed.query or parsed.fragment or "%" in parsed.path or "\\" in parsed.path
            or not parsed.path.startswith("/") or "//" in parsed.path
            or any(part in (".", "..") for part in parsed.path.split("/"))):
        raise ValueError("URL must be canonical HTTPS without credentials, port or encoded path")


def bounded_fetch(url: str, *, max_bytes: int, deadline: LiveDeadline,
                  allowed_urls: frozenset[str], opener: Callable | None = None,
                  utc_now: Callable[[], datetime] = _utc_now) -> FetchedBytes:
    """Fetch one exact allowlisted URL once; no redirects, retries or chunking.

    The injected opener is a trusted offline-test seam. Production defaults to
    a redirect-rejecting urllib handler. Read1 is used only for fixed-length,
    unencoded bodies; each read resets the underlying inactivity timeout.
    """
    _safe_url(url)
    if not isinstance(allowed_urls, frozenset) or url not in allowed_urls:
        raise ValueError("URL is outside the exact allowlist")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 32_000_000:
        raise ValueError("max_bytes must be in [1, 32000000]")
    request = Request(url, headers={"User-Agent": "Analytics-Lab-public-live/1.0",
                                   "Accept-Encoding": "identity", "Cache-Control": "no-cache"})
    with (opener or _open)(request, timeout=deadline.timeout()) as response:
        deadline.check()
        if response.geturl() != url or getattr(response, "status", None) != 200:
            raise RuntimeError("public-live requires exact URL and HTTP 200")
        length = response.headers.get("Content-Length")
        if (type(length) is not str or re.fullmatch(r"[1-9][0-9]{0,8}", length) is None
                or int(length) > max_bytes or response.headers.get("Transfer-Encoding") is not None
                or response.headers.get("Content-Encoding") not in (None, "identity")):
            raise RuntimeError("public-live requires a bounded exact Content-Length without encoding")
        http_date = response.headers.get("Date")
        if http_date is not None and (type(http_date) is not str or len(http_date) > 128
                                      or any(not 32 <= ord(c) < 127 for c in http_date)):
            raise RuntimeError("invalid HTTP Date metadata")
        sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        if not callable(getattr(sock, "settimeout", None)) or not callable(getattr(response, "read1", None)):
            raise RuntimeError("bounded HTTP body interface unavailable")
        expected, total = int(length), 0
        blocks: list[bytes] = []
        digest = hashlib.sha256()
        while total < expected:
            sock.settimeout(deadline.timeout())
            requested = min(64 * 1024, expected - total)
            block = response.read1(requested)
            deadline.check()
            if type(block) is not bytes or not block or len(block) > requested:
                raise RuntimeError("public-live body is truncated or exceeds its declared length")
            total += len(block)
            digest.update(block)
            blocks.append(block)
        # A real HTTPResponse closes its socket on its final declared byte.
        # The entity is complete; an additional settimeout/read would be wrong.
    receipt = utc_now()
    if not isinstance(receipt, datetime) or receipt.tzinfo is None or receipt.utcoffset() is None:
        raise ValueError("receipt UTC clock must return an aware datetime")
    deadline.check()
    return FetchedBytes(b"".join(blocks), digest.hexdigest(), total, http_date,
                        receipt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate catalog JSON key")
        result[key] = value
    return result


def _verify_catalog(payload: bytes) -> None:
    try:
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
        entries = document["data"]
        if not isinstance(entries, list) or len(entries) > 5000:
            raise ValueError("invalid catalog rows")
        selected = [row["cctv"] for row in entries
                    if isinstance(row, dict) and isinstance(row.get("cctv"), dict)
                    and row["cctv"].get("index") == "88"]
        if len(selected) != 1:
            raise ValueError("selected camera must be unique")
        camera = selected[0]
        if (camera["inService"] != "true" or camera["location"]["district"] != "1"
                or camera["location"]["locationName"] != CAMERA_NAME
                or camera["imageData"]["streamingVideoURL"] != MASTER_URL):
            raise ValueError("selected camera changed or is out of service")
    except (UnicodeError, ValueError, KeyError, TypeError, RecursionError) as exc:
        raise RuntimeError("Caltrans catalog camera identity mismatch") from exc


def _lines(payload: bytes) -> list[str]:
    try:
        text = payload.decode("ascii")
    except UnicodeError as exc:
        raise RuntimeError("playlist must be ASCII") from exc
    if "\r" in text.replace("\r\n", "") or any(ord(c) < 32 and c not in "\r\n" for c in text):
        raise RuntimeError("playlist contains unsupported control characters")
    lines = text.splitlines()
    if not lines or lines[0] != "#EXTM3U" or any(not line for line in lines):
        raise RuntimeError("playlist must have a canonical EXTM3U header and nonempty lines")
    return lines


def _master(payload: bytes) -> tuple[str, str]:
    lines = _lines(payload)
    if (len(lines) != 4 or lines[1] != "#EXT-X-VERSION:3"
            or not lines[2].startswith("#EXT-X-STREAM-INF:")):
        raise RuntimeError("master must contain exactly the reviewed single 320x240 variant")
    attributes: dict[str, str] = {}
    for attribute in lines[2].removeprefix("#EXT-X-STREAM-INF:").split(","):
        key, separator, value = attribute.partition("=")
        if not separator or key not in {"BANDWIDTH", "RESOLUTION", "CODECS"} or key in attributes:
            raise RuntimeError("master variant has duplicate or unsupported attributes")
        attributes[key] = value
    if (set(attributes) != {"BANDWIDTH", "RESOLUTION", "CODECS"}
            or attributes["RESOLUTION"] != "320x240" or attributes["CODECS"] != '"avc1.100.41"'
            or re.fullmatch(r"[1-9][0-9]{0,5}", attributes["BANDWIDTH"]) is None
            or int(attributes["BANDWIDTH"]) > 500_000):
        raise RuntimeError("master variant exceeds the reviewed codec/dimensions/bandwidth")
    match = re.fullmatch(r"chunklist_w([0-9]{1,19})\.m3u8", lines[3])
    if match is None:
        raise RuntimeError("master child URI is outside the reviewed directory/session pattern")
    return urljoin(MASTER_URL, lines[3]), match[1]


def _media(payload: bytes, session: str) -> tuple[int, tuple[tuple[int, float, str], ...]]:
    lines = _lines(payload)
    fields: dict[str, int] = {}
    segments: list[tuple[int, float, str]] = []
    index = 1
    allowed = {"#EXT-X-VERSION", "#EXT-X-TARGETDURATION", "#EXT-X-MEDIA-SEQUENCE",
               "#EXT-X-DISCONTINUITY-SEQUENCE"}
    while index < len(lines):
        line = lines[index]
        if line.startswith("#EXTINF:"):
            match = re.fullmatch(r"#EXTINF:([0-9]{1,2}(?:\.[0-9]{1,6})?),", line)
            if match is None or index + 1 >= len(lines) or len(segments) == 3:
                raise RuntimeError("unsupported segment duration/count")
            duration = float(match[1])
            if not 0 < duration <= 15:
                raise RuntimeError("segment duration exceeds live-window bound")
            uri = lines[index + 1]
            name = re.fullmatch(r"media_w([0-9]{1,19})_([0-9]{1,19})\.ts", uri)
            if name is None or name[1] != session:
                raise RuntimeError("segment URI escaped the reviewed session/directory")
            segments.append((int(name[2]), duration, urljoin(MASTER_URL, uri)))
            index += 2
            continue
        key, separator, value = line.partition(":")
        if (segments or key not in allowed or key in fields or not separator
                or re.fullmatch(r"0|[1-9][0-9]{0,18}", value) is None):
            raise RuntimeError("unsupported, duplicate or out-of-order HLS tag")
        fields[key] = int(value)
        index += 1
    if (set(fields) != allowed or fields["#EXT-X-VERSION"] not in (3, 4)
            or not 1 <= fields["#EXT-X-TARGETDURATION"] <= 15
            or fields["#EXT-X-DISCONTINUITY-SEQUENCE"] != 0 or len(segments) != 3):
        raise RuntimeError("unsupported media playlist scope")
    sequence = fields["#EXT-X-MEDIA-SEQUENCE"]
    if (sequence > 2**63 - 3 or tuple(item[0] for item in segments) != tuple(range(sequence, sequence + 3))
            or any(duration > fields["#EXT-X-TARGETDURATION"] for _, duration, _ in segments)
            or sum(item[1] for item in segments) > 35):
        raise RuntimeError("playlist sequence or total duration exceeds the live-window contract")
    return sequence, tuple(segments)


def _identity(url: str, fetched: FetchedBytes) -> dict[str, Any]:
    return {"url": url, "sha256": fetched.sha256, "size_bytes": fetched.size_bytes,
            "http_date": fetched.http_date, "received_utc": fetched.received_utc}


def capture_live_window(work_dir: Path, *, opener: Callable | None = None,
                        clock: Callable[[], float] = time.monotonic,
                        sleep: Callable[[float], None] = time.sleep,
                        utc_now: Callable[[], datetime] = _utc_now,
                        deadline: LiveDeadline | None = None) -> LiveWindow:
    """Acquire one three-segment window with exactly seven GETs and no retry."""
    root_value = os.environ.get("RUNNER_TEMP")
    if not root_value or not isinstance(work_dir, Path):
        raise ValueError("an existing work directory below RUNNER_TEMP is required")
    root, destination = Path(root_value).absolute(), work_dir.absolute()
    if (not root.is_dir() or root.resolve() != root or not destination.is_dir()
            or destination.resolve() != destination or root not in destination.parents
            or any(destination.iterdir())):
        raise ValueError("work directory must be new, empty, canonical and below RUNNER_TEMP")
    budget = deadline or LiveDeadline(clock)
    def fetch(url: str, cap: int) -> FetchedBytes:
        return bounded_fetch(url, max_bytes=cap, deadline=budget, allowed_urls=frozenset({url}),
                             opener=opener, utc_now=utc_now)
    catalog = fetch(CATALOG_URL, _CATALOG_CAP)
    _verify_catalog(catalog.payload)
    master = fetch(MASTER_URL, _PLAYLIST_CAP)
    child_url, session = _master(master.payload)
    child = fetch(child_url, _PLAYLIST_CAP)
    first_child_time = budget.clock()
    sequence, planned = _media(child.payload, session)
    downloaded: list[tuple[int, float, str, FetchedBytes]] = []
    total = 0
    for number, duration, url in planned:
        remaining = _TOTAL_SEGMENT_CAP - total
        if remaining <= 0:
            raise RuntimeError("total TS byte budget exhausted")
        payload = fetch(url, min(_SEGMENT_CAP, remaining))
        if payload.size_bytes % 188 or any(value != 0x47 for value in payload.payload[::188]):
            raise RuntimeError("segment is not packet-aligned sync-checked MPEG-TS")
        downloaded.append((number, duration, url, payload))
        total += payload.size_bytes
    delay = max(0.0, first_child_time + 15.0 - budget.clock())
    if delay:
        if budget.timeout(15.0) < delay:
            raise TimeoutError("insufficient budget for the single advancement check")
        sleep(delay)
    budget.check()
    advanced = fetch(child_url, _PLAYLIST_CAP)
    advanced_sequence, _ = _media(advanced.payload, session)
    if advanced_sequence <= sequence:
        raise RuntimeError("live playlist did not advance; inference is prohibited")
    metadata = {
        "schema": "caltrans-public-live-window-v1", "source_id": "caltrans-d1-cctv-88",
        "camera_name": CAMERA_NAME, "camera_capture_utc": None,
        "catalog": _identity(CATALOG_URL, catalog), "master": _identity(MASTER_URL, master),
        "initial_child": _identity(child_url, child), "advanced_child": _identity(child_url, advanced),
        "initial_media_sequence": sequence, "advanced_media_sequence": advanced_sequence,
        "sequence_advanced": True, "freshness_wait_seconds": delay,
        "declared_duration_seconds": sum(item[1] for item in planned),
        "segment_bytes": total, "source_get_count": 7,
        "segments": [{**_identity(url, payload), "sequence": number, "duration_seconds": duration}
                     for number, duration, url, payload in downloaded],
    }
    created: list[Path] = []
    try:
        segments: list[LiveSegment] = []
        for number, duration, _, payload in downloaded:
            budget.check()
            path = destination / f"segment-{number}.ts"
            with path.open("xb") as handle:
                created.append(path)
                handle.write(payload.payload)
            segments.append(LiveSegment(number, duration, path, payload.sha256, payload.size_bytes))
        budget.check()
        return LiveWindow(tuple(segments), metadata)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
