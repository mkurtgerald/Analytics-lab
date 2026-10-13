"""Metadata-only producer export of real TrackingSession outputs (schema v1).

The producer does not assign authenticated tenant/site scope, verify detector
accuracy, assert model/data redistribution rights, or grant product permissions.
A consumer must bind source/stream to its own authenticated session, persist
replay/dedup state, and handle unavailable frames as unknown rather than empty.
Track IDs identify associations only inside the named tracking session.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable

from .tracking import NormalizedBox, TrackedDetection

SCHEMA = "analytics.tracked-frame.v1"
_MAX_TIMESTAMP_MS = 253402300799999
_MAX_TRACKS = 512
_MAX_JSON_BYTES = 262_144
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REASONS = frozenset({
    "source_unavailable", "detector_unavailable", "tracker_unavailable", "budget_exceeded",
})
_SEMANTICS = {
    "coordinates": "normalized_xyxy",
    "confidence": "uncalibrated_model_score",
    "track_scope": "source_stream_session_local",
    "identity_claim": False,
    "authorizes_action": False,
    "source_scope_authorized": False,
}
_CONTEXT_FIELDS = {
    "event_id", "source_ref", "stream_ref", "session_id", "frame_index",
    "utc_timestamp_ms", "media_pts", "timebase_num", "timebase_den",
    "producer", "producer_revision", "config_sha256", "model_sha256", "model_revision", "source_revision",
}
_FRAME_FIELDS = {
    "source_ref", "stream_ref", "session_id", "frame_index",
    "utc_timestamp_ms", "media_pts", "timebase_num", "timebase_den",
}
_PRODUCER_FIELDS = {"name", "revision", "config_sha256", "model_sha256", "model_revision", "source_revision"}
_TOP_FIELDS = {"schema", "event_id", "frame", "producer", "semantics", "status", "unavailable_reason", "tracks"}
_TRACK_FIELDS = {"track_id", "category", "confidence", "box", "model_class_id"}


def _token(value: object, name: str) -> str:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must be an opaque 1-128 character token")
    return value


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _closed(value: object, fields: set[str], name: str) -> dict:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} has missing or unexpected fields")
    return value


def _snapshot_track(item: object) -> TrackedDetection:
    if not isinstance(item, TrackedDetection):
        raise ValueError("tracks must contain TrackedDetection objects")
    if not isinstance(item.box, NormalizedBox):
        raise ValueError("track box must be NormalizedBox")
    # Frozen dataclasses can still be changed using object.__setattr__ by a donor.
    # Reconstruct all values before emitting them and detach from donor-owned data.
    box = NormalizedBox(item.box.x_min, item.box.y_min, item.box.x_max, item.box.y_max)
    _token(item.track_id, "track_id")
    return TrackedDetection(item.track_id, item.category, item.confidence, box, item.model_class_id)


@dataclass(frozen=True)
class TrackingFrameContext:
    event_id: str
    source_ref: str
    stream_ref: str
    session_id: str
    frame_index: int
    utc_timestamp_ms: int
    producer: str
    producer_revision: str
    config_sha256: str
    model_sha256: str | None = None
    model_revision: str | None = None
    source_revision: str | None = None
    media_pts: int | None = None
    timebase_num: int | None = None
    timebase_den: int | None = None

    def __post_init__(self) -> None:
        for field in ("event_id", "source_ref", "stream_ref", "session_id", "producer", "producer_revision"):
            _token(getattr(self, field), field)
        if self.source_revision is not None:
            _token(self.source_revision, "source_revision")
        _digest(self.config_sha256, "config_sha256")
        if self.model_sha256 is not None:
            _digest(self.model_sha256, "model_sha256")
        if self.model_revision is not None:
            _token(self.model_revision, "model_revision")
            if self.model_sha256 is None:
                raise ValueError("model_revision requires an immutable model digest")
        if type(self.frame_index) is not int or not 0 <= self.frame_index <= 2**63 - 1:
            raise ValueError("frame_index must be a nonnegative signed-64-bit integer")
        if type(self.utc_timestamp_ms) is not int or not 0 <= self.utc_timestamp_ms <= _MAX_TIMESTAMP_MS:
            raise ValueError("utc_timestamp_ms is outside the supported UTC range")
        if self.media_pts is None:
            if self.timebase_num is not None or self.timebase_den is not None:
                raise ValueError("media timebase requires media_pts")
        else:
            if type(self.media_pts) is not int or not -(2**63) <= self.media_pts < 2**63:
                raise ValueError("media_pts must be a signed-64-bit integer")
            for field in ("timebase_num", "timebase_den"):
                value = getattr(self, field)
                if type(value) is not int or not 1 <= value <= 1_000_000_000:
                    raise ValueError(f"{field} must be a positive bounded integer")


@dataclass(frozen=True)
class TrackingFrameEnvelope:
    context: TrackingFrameContext
    tracks: tuple[TrackedDetection, ...] = ()
    status: str = "ok"
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, TrackingFrameContext):
            raise ValueError("context must be TrackingFrameContext")
        self.context.__post_init__()
        if self.status not in ("ok", "unavailable"):
            raise ValueError("unsupported tracking frame status")
        if self.status == "ok" and self.unavailable_reason is not None:
            raise ValueError("available frames cannot carry an unavailable reason")
        if self.status == "unavailable" and (type(self.unavailable_reason) is not str
                                             or self.unavailable_reason not in _REASONS):
            raise ValueError("unavailable frames require a supported reason")
        try:
            iterator = iter(self.tracks)
        except TypeError as exc:
            raise ValueError("tracks must be iterable") from exc
        copied: list[TrackedDetection] = []
        ids: set[str] = set()
        for item in iterator:
            if len(copied) >= _MAX_TRACKS:
                raise ValueError("tracking frame exceeds track bound")
            track = _snapshot_track(item)
            if track.track_id in ids:
                raise ValueError("duplicate track_id in tracking frame")
            ids.add(track.track_id)
            copied.append(track)
        if self.status == "unavailable" and copied:
            raise ValueError("unavailable frames cannot report tracks")
        object.__setattr__(self, "tracks", tuple(copied))

    def to_json(self) -> str:
        # Revalidate (and detach) after potential object.__setattr__ mutation.
        snap = TrackingFrameEnvelope(self.context, self.tracks, self.status, self.unavailable_reason)
        c = snap.context
        payload = {
            "schema": SCHEMA,
            "event_id": c.event_id,
            "frame": {name: getattr(c, name) for name in _FRAME_FIELDS},
            "producer": {
                "name": c.producer, "revision": c.producer_revision,
                "config_sha256": c.config_sha256, "model_sha256": c.model_sha256,
                "model_revision": c.model_revision, "source_revision": c.source_revision,
            },
            "semantics": _SEMANTICS.copy(),
            "status": snap.status,
            "unavailable_reason": snap.unavailable_reason,
            "tracks": [{
                "track_id": item.track_id, "category": item.category,
                "confidence": item.confidence, "model_class_id": item.model_class_id,
                "box": [item.box.x_min, item.box.y_min, item.box.x_max, item.box.y_max],
            } for item in snap.tracks],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        if len(raw.encode("utf-8")) > _MAX_JSON_BYTES:
            raise ValueError("tracking frame JSON exceeds size bound")
        return raw


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    obj: dict = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON field")
        obj[key] = value
    return obj


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is prohibited: {value}")


def parse_tracking_frame_json(raw: str) -> TrackingFrameEnvelope:
    """Decode exact schema v1; metadata is untrusted until consumer authorizes it."""
    if type(raw) is not str:
        raise ValueError("tracking frame must be JSON text")
    try:
        if len(raw.encode("utf-8")) > _MAX_JSON_BYTES:
            raise ValueError("tracking frame JSON exceeds size bound")
        payload = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError, OverflowError) as exc:
        raise ValueError("invalid tracking frame JSON") from exc
    _closed(payload, _TOP_FIELDS, "tracking frame")
    if payload["schema"] != SCHEMA:
        raise ValueError("unsupported tracking frame schema")
    if payload["semantics"] != _SEMANTICS:
        raise ValueError("unsupported or authorizing tracking semantics")
    frame = _closed(payload["frame"], _FRAME_FIELDS, "frame")
    producer = _closed(payload["producer"], _PRODUCER_FIELDS, "producer")
    context = TrackingFrameContext(
        event_id=payload["event_id"], **frame,
        producer=producer["name"], producer_revision=producer["revision"],
        config_sha256=producer["config_sha256"], model_sha256=producer["model_sha256"],
        model_revision=producer["model_revision"], source_revision=producer["source_revision"],
    )
    tracks = payload["tracks"]
    if type(tracks) is not list or len(tracks) > _MAX_TRACKS:
        raise ValueError("tracks must be a bounded JSON list")
    decoded: list[TrackedDetection] = []
    for item in tracks:
        value = _closed(item, _TRACK_FIELDS, "track")
        coords = value["box"]
        if type(coords) is not list or len(coords) != 4:
            raise ValueError("normalized box requires four coordinates")
        decoded.append(TrackedDetection(
            value["track_id"], value["category"], value["confidence"],
            NormalizedBox(*coords), value["model_class_id"],
        ))
    return TrackingFrameEnvelope(context, tuple(decoded), payload["status"], payload["unavailable_reason"])
