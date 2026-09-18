"""Model-independent person-down persistence rule; no video inference is included.

Inputs are observations from a future, separately validated perception adapter.
An event asserts only that down-posture observations persisted. It does not
establish a fall, injury, intent, negligence, or cause. All events are candidates.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import re
from uuid import NAMESPACE_URL, uuid5


_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_MAX_TIMESTAMP_MS = 253402300799999


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be an opaque 1-128 character identifier")


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _score(value: float, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _iso(timestamp_ms: int) -> str:
    instant = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=timestamp_ms)
    return instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Observation:
    timestamp_ms: int
    track_id: str
    posture: str
    confidence: float

    def __post_init__(self) -> None:
        _integer(self.timestamp_ms, "timestamp_ms")
        if self.timestamp_ms > _MAX_TIMESTAMP_MS:
            raise ValueError("timestamp_ms exceeds the supported UTC date range")
        _identifier(self.track_id, "track_id")
        if self.posture not in ("upright", "down", "other", "unknown"):
            raise ValueError("unsupported posture")
        _score(self.confidence, "confidence")


@dataclass(frozen=True)
class Config:
    down_duration_ms: int = 3000
    max_gap_ms: int = 750
    min_samples: int = 4
    min_confidence: float = 0.7
    max_unknown_gap_ms: int = 0
    track_ttl_ms: int = 30000
    max_tracks: int = 1024

    def __post_init__(self) -> None:
        for name in ("down_duration_ms", "max_gap_ms", "track_ttl_ms", "max_tracks"):
            _integer(getattr(self, name), name, 1)
        _integer(self.min_samples, "min_samples", 2)
        _score(self.min_confidence, "min_confidence")
        _integer(self.max_unknown_gap_ms, "max_unknown_gap_ms")
        if self.max_unknown_gap_ms > self.max_gap_ms:
            raise ValueError("max_unknown_gap_ms cannot exceed max_gap_ms")
        if self.track_ttl_ms <= self.max_gap_ms:
            raise ValueError("track_ttl_ms must exceed max_gap_ms")


@dataclass
class _Track:
    last_timestamp_ms: int
    start_timestamp_ms: int | None = None
    min_confidence: float = 1.0
    count: int = 0
    emitted: bool = False
    last_down_timestamp_ms: int | None = None
    unknown_since_down: bool = False

    def reset_run(self) -> None:
        self.start_timestamp_ms = None
        self.min_confidence = 1.0
        self.count = 0
        self.emitted = False
        self.last_down_timestamp_ms = None
        self.unknown_since_down = False


class PersonDownEngine:
    """One ordered source/session; temporary track IDs never represent identity.

    Supply nondecreasing timestamps across the source and strictly increasing
    timestamps for each track. A new stream/tracker session needs a new engine
    and session_id. Do not reuse a track ID for a different person within its
    TTL. Calls are synchronous and not thread-safe. Track count is bounded.

    By default, every non-down posture breaks persistence. A caller may
    explicitly set ``max_unknown_gap_ms`` to bridge a short run of ``unknown``
    observations only. ``upright``, ``other`` and low-confidence ``down`` still
    reset immediately, and an unknown span is bounded from the last accepted
    down observation to the next accepted down observation.
    """

    def __init__(self, source_id: str, session_id: str, config: Config | None = None):
        _identifier(source_id, "source_id")
        _identifier(session_id, "session_id")
        if config is not None and not isinstance(config, Config):
            raise ValueError("config must be a Config")
        self.source_id = source_id
        self.session_id = session_id
        self.config = config or Config()
        self._tracks: dict[str, _Track] = {}
        self._watermark = -1

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def observe(self, observation: Observation) -> list[dict]:
        if not isinstance(observation, Observation):
            raise ValueError("observation must be an Observation")
        t = observation.timestamp_ms
        if t < self._watermark:
            raise ValueError("out-of-order source timestamp")
        track = self._tracks.get(observation.track_id)
        if track is not None and t <= track.last_timestamp_ms:
            raise ValueError("duplicate or out-of-order track timestamp")

        # Expiration breaks continuity; no recovery or safety inference is made.
        expired = [key for key, item in self._tracks.items()
                   if t - item.last_timestamp_ms > self.config.track_ttl_ms]
        for key in expired:
            del self._tracks[key]
        track = self._tracks.get(observation.track_id)
        if track is None:
            if len(self._tracks) >= self.config.max_tracks:
                raise RuntimeError("track capacity exceeded; no observation accepted")
            track = _Track(t)
            self._tracks[observation.track_id] = track
        elif t - track.last_timestamp_ms > self.config.max_gap_ms:
            track.reset_run()

        self._watermark = t
        track.last_timestamp_ms = t

        if observation.posture == "unknown":
            if (
                self.config.max_unknown_gap_ms > 0
                and track.start_timestamp_ms is not None
                and track.last_down_timestamp_ms is not None
                and t - track.last_down_timestamp_ms <= self.config.max_unknown_gap_ms
            ):
                track.unknown_since_down = True
                return []
            track.reset_run()
            return []

        if observation.posture != "down" or observation.confidence < self.config.min_confidence:
            # Conflicting or low-confidence observations cannot extend a run.
            track.reset_run()
            return []

        if (
            track.unknown_since_down
            and track.last_down_timestamp_ms is not None
            and t - track.last_down_timestamp_ms > self.config.max_unknown_gap_ms
        ):
            # The next accepted down sample arrived beyond the bounded unknown span.
            track.reset_run()

        if track.start_timestamp_ms is None:
            track.start_timestamp_ms = t
        track.count += 1
        track.min_confidence = min(track.min_confidence, observation.confidence)
        track.last_down_timestamp_ms = t
        track.unknown_since_down = False
        if (not track.emitted
                and t - track.start_timestamp_ms >= self.config.down_duration_ms
                and track.count >= self.config.min_samples):
            track.emitted = True
            return [self._event(observation, track)]
        return []

    def _event(self, observation: Observation, track: _Track) -> dict:
        start = track.start_timestamp_ms
        assert start is not None
        identity = json.dumps(["person_down/0.0.1", self.source_id, self.session_id,
                               observation.track_id, start], separators=(",", ":"))
        return {
            "schema_version": "0.1",
            "event_id": str(uuid5(NAMESPACE_URL, identity)),
            "analytic_id": "person_down_persistence",
            "event_type": "person_down_candidate",
            "source_id": self.source_id,
            "start_time": _iso(start),
            "end_time": _iso(observation.timestamp_ms),
            "confidence": track.min_confidence,
            "severity": "review",
            "status": "candidate",
            "objects": [{"type": "person", "track_id": observation.track_id}],
            "zones": [],
            "observations": [{"type": "down_posture_persistence", "sample_count": track.count,
                              "start_time": _iso(start), "end_time": _iso(observation.timestamp_ms)}],
            "evidence": [{"type": "observation_window", "source_id": self.source_id,
                          "session_id": self.session_id, "track_id": observation.track_id,
                          "start_timestamp_ms": start, "end_timestamp_ms": observation.timestamp_ms}],
            "relationships": [],
            "recommended_actions": [],
            "metadata": {
                "analytic_version": "0.0.1",
                "confidence_basis": "minimum_input_score_not_calibrated_probability",
                "maturity": "synthetic_regression_only",
                "inferences_not_supported": ["fall", "injury", "cause", "fault", "intent"],
                "thresholds": {"down_duration_ms": self.config.down_duration_ms,
                               "max_gap_ms": self.config.max_gap_ms,
                               "min_samples": self.config.min_samples,
                               "min_confidence": self.config.min_confidence,
                               "max_unknown_gap_ms": self.config.max_unknown_gap_ms},
            },
        }
