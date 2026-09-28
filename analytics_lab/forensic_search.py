"""Bounded deterministic forensic-search foundation.

This module indexes only normalized analytic metadata already produced by
reviewed Analytics Lab contracts. It stores no image/video pixels, performs no
biometric identity matching, and makes no cross-camera identity claim.

The first search surface supports time-window, camera/source, category,
session-local track, normalized plate/OCR text, and exact generic analytic
attribute filters with deterministic ordering and bounded result counts.

Appearance/vector similarity remains a separate future donor-backed capability.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .lpr_ocr import LPRObservation
from .tracking import NormalizedBox, TrackedDetection

_MAX_TIMESTAMP_MS = 253402300799999
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.:/-]{1,128}\Z")
_SAFE_ATTR = re.compile(r"[A-Za-z0-9_.:/ -]{1,128}\Z")
_SAFE_PLATE = re.compile(r"[A-Z0-9]{1,16}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _token(value: object, name: str) -> str:
    if not isinstance(value, str) or _SAFE_TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


def _attr_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = " ".join(value.strip().split())
    if _SAFE_ATTR.fullmatch(normalized) is None:
        raise ValueError(f"{name} is invalid")
    return normalized


def _timestamp(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_TIMESTAMP_MS:
        raise ValueError(f"{name} is outside the supported UTC range")
    return value


def _confidence(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


@dataclass(frozen=True, order=True)
class ForensicAttribute:
    """One deterministic searchable facet emitted by a reviewed analytic."""

    name: str
    value: str
    confidence: float
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _attr_text(self.name, "attribute name"))
        object.__setattr__(self, "value", _attr_text(self.value, "attribute value"))
        object.__setattr__(
            self,
            "confidence",
            _confidence(self.confidence, "attribute confidence"),
        )
        object.__setattr__(self, "provenance", _token(self.provenance, "attribute provenance"))


@dataclass(frozen=True)
class ForensicEvidenceLink:
    """Immutable provenance for one searchable analytic observation."""

    event_id: str
    producer: str
    producer_version: str
    config_sha256: str
    model_sha256: str | None = None
    source_revision: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _token(self.event_id, "evidence event_id"))
        object.__setattr__(self, "producer", _token(self.producer, "evidence producer"))
        object.__setattr__(
            self,
            "producer_version",
            _token(self.producer_version, "evidence producer_version"),
        )
        if not isinstance(self.config_sha256, str) or _SHA256.fullmatch(self.config_sha256) is None:
            raise ValueError("config_sha256 must be lowercase SHA-256")
        if self.model_sha256 is not None:
            if not isinstance(self.model_sha256, str) or _SHA256.fullmatch(self.model_sha256) is None:
                raise ValueError("model_sha256 must be lowercase SHA-256")
        if self.source_revision is not None:
            object.__setattr__(
                self,
                "source_revision",
                _token(self.source_revision, "evidence source_revision"),
            )


@dataclass(frozen=True)
class ForensicRecord:
    """One searchable normalized analytic observation.

    track_id remains session-local association metadata only. It must never be
    interpreted as biometric identity or a cross-camera person identity.
    """

    observation_id: str
    source_id: str
    timestamp_ms: int
    category: str
    confidence: float
    box: NormalizedBox | None = None
    track_id: str | None = None
    plate_text: str | None = None
    attributes: tuple[ForensicAttribute, ...] = ()
    evidence: ForensicEvidenceLink | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", _token(self.observation_id, "observation_id"))
        object.__setattr__(self, "source_id", _token(self.source_id, "source_id"))
        object.__setattr__(self, "timestamp_ms", _timestamp(self.timestamp_ms, "timestamp_ms"))
        object.__setattr__(self, "category", _attr_text(self.category, "category"))
        object.__setattr__(self, "confidence", _confidence(self.confidence, "confidence"))

        if self.box is not None and not isinstance(self.box, NormalizedBox):
            raise ValueError("box must be NormalizedBox or None")

        if self.track_id is not None:
            object.__setattr__(self, "track_id", _token(self.track_id, "track_id"))

        if self.plate_text is not None:
            if not isinstance(self.plate_text, str) or _SAFE_PLATE.fullmatch(self.plate_text) is None:
                raise ValueError("plate_text must be normalized A-Z/0-9")
            if self.category != "license_plate":
                raise ValueError("plate_text is only valid for license_plate records")

        if not isinstance(self.attributes, tuple):
            raise ValueError("attributes must be an immutable tuple")
        if len(self.attributes) > 64:
            raise ValueError("attribute count exceeds supported bound")
        if any(not isinstance(item, ForensicAttribute) for item in self.attributes):
            raise ValueError("attributes must contain ForensicAttribute values")
        keys = [(item.name, item.value, item.provenance) for item in self.attributes]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate forensic attribute")
        if self.evidence is not None and not isinstance(self.evidence, ForensicEvidenceLink):
            raise ValueError("evidence must be ForensicEvidenceLink or None")


@dataclass(frozen=True)
class AttributeFilter:
    name: str
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _attr_text(self.name, "attribute filter name"))
        object.__setattr__(self, "value", _attr_text(self.value, "attribute filter value"))


@dataclass(frozen=True)
class ForensicQuery:
    start_ms: int | None = None
    end_ms: int | None = None
    source_ids: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    track_ids: tuple[str, ...] = ()
    plate_text: str | None = None
    plate_prefix: str | None = None
    attributes: tuple[AttributeFilter, ...] = ()
    min_confidence: float = 0.0
    limit: int = 100

    def __post_init__(self) -> None:
        if self.start_ms is not None:
            object.__setattr__(self, "start_ms", _timestamp(self.start_ms, "start_ms"))
        if self.end_ms is not None:
            object.__setattr__(self, "end_ms", _timestamp(self.end_ms, "end_ms"))
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")

        for field_name in ("source_ids", "categories", "track_ids"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise ValueError(f"{field_name} must be an immutable tuple")
            if len(values) > 256:
                raise ValueError(f"{field_name} exceeds supported bound")
            normalized = tuple(
                _token(item, field_name)
                if field_name != "categories"
                else _attr_text(item, "category filter")
                for item in values
            )
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{field_name} contains duplicates")
            object.__setattr__(self, field_name, normalized)

        if self.plate_text is not None:
            if not isinstance(self.plate_text, str) or _SAFE_PLATE.fullmatch(self.plate_text) is None:
                raise ValueError("plate_text filter must be normalized A-Z/0-9")
        if self.plate_prefix is not None:
            if not isinstance(self.plate_prefix, str) or _SAFE_PLATE.fullmatch(self.plate_prefix) is None:
                raise ValueError("plate_prefix filter must be normalized A-Z/0-9")
        if self.plate_text is not None and self.plate_prefix is not None:
            raise ValueError("use plate_text or plate_prefix, not both")

        if not isinstance(self.attributes, tuple):
            raise ValueError("attributes must be an immutable tuple")
        if len(self.attributes) > 32:
            raise ValueError("attribute filter count exceeds supported bound")
        if any(not isinstance(item, AttributeFilter) for item in self.attributes):
            raise ValueError("attributes must contain AttributeFilter values")

        object.__setattr__(
            self,
            "min_confidence",
            _confidence(self.min_confidence, "min_confidence"),
        )
        if type(self.limit) is not int or not 1 <= self.limit <= 1000:
            raise ValueError("limit must be an integer in [1, 1000]")


class ForensicIndex:
    """Small bounded deterministic metadata index."""

    def __init__(self, *, max_records: int = 100_000) -> None:
        if type(max_records) is not int or not 1 <= max_records <= 1_000_000:
            raise ValueError("max_records must be an integer in [1, 1000000]")
        self.max_records = max_records
        self._records: dict[str, ForensicRecord] = {}

    def add(self, record: ForensicRecord) -> None:
        if not isinstance(record, ForensicRecord):
            raise ValueError("record must be ForensicRecord")
        if record.evidence is None:
            raise ValueError("searchable forensic record requires immutable evidence provenance")
        existing = self._records.get(record.observation_id)
        if existing is not None:
            if existing != record:
                raise ValueError("observation_id conflicts with existing record")
            return
        if len(self._records) >= self.max_records:
            raise RuntimeError("forensic index capacity exceeded")
        self._records[record.observation_id] = record

    def extend(self, records: Iterable[ForensicRecord]) -> None:
        try:
            values = tuple(records)
        except TypeError as exc:
            raise ValueError("records must be iterable") from exc
        if len(values) > self.max_records:
            raise RuntimeError("input batch exceeds forensic index capacity")
        for item in values:
            self.add(item)

    def search(self, query: ForensicQuery) -> tuple[ForensicRecord, ...]:
        if not isinstance(query, ForensicQuery):
            raise ValueError("query must be ForensicQuery")
        source_ids = set(query.source_ids)
        categories = set(query.categories)
        track_ids = set(query.track_ids)
        required_attributes = {(item.name, item.value) for item in query.attributes}

        result: list[ForensicRecord] = []
        for record in self._records.values():
            if query.start_ms is not None and record.timestamp_ms < query.start_ms:
                continue
            if query.end_ms is not None and record.timestamp_ms > query.end_ms:
                continue
            if source_ids and record.source_id not in source_ids:
                continue
            if categories and record.category not in categories:
                continue
            if track_ids and record.track_id not in track_ids:
                continue
            if record.confidence < query.min_confidence:
                continue
            if query.plate_text is not None and record.plate_text != query.plate_text:
                continue
            if query.plate_prefix is not None:
                if record.plate_text is None or not record.plate_text.startswith(query.plate_prefix):
                    continue
            if required_attributes:
                present = {(item.name, item.value) for item in record.attributes}
                if not required_attributes.issubset(present):
                    continue
            result.append(record)

        result.sort(key=lambda item: (item.timestamp_ms, item.source_id, item.observation_id))
        return tuple(result[: query.limit])


def record_from_track(
    *,
    observation_id: str,
    source_id: str,
    timestamp_ms: int,
    tracked: TrackedDetection,
    evidence: ForensicEvidenceLink,
    attributes: tuple[ForensicAttribute, ...] = (),
) -> ForensicRecord:
    if not isinstance(tracked, TrackedDetection):
        raise ValueError("tracked must be TrackedDetection")
    return ForensicRecord(
        observation_id=observation_id,
        source_id=source_id,
        timestamp_ms=timestamp_ms,
        category=tracked.category,
        confidence=tracked.confidence,
        box=tracked.box,
        track_id=tracked.track_id,
        attributes=attributes,
        evidence=evidence,
    )


def record_from_lpr(
    *,
    observation_id: str,
    source_id: str,
    timestamp_ms: int,
    observation: LPRObservation,
    evidence: ForensicEvidenceLink,
    attributes: tuple[ForensicAttribute, ...] = (),
) -> ForensicRecord:
    if not isinstance(observation, LPRObservation):
        raise ValueError("observation must be LPRObservation")
    plate = observation.plate
    return ForensicRecord(
        observation_id=observation_id,
        source_id=source_id,
        timestamp_ms=timestamp_ms,
        category="license_plate",
        confidence=min(plate.confidence, observation.text.confidence),
        box=NormalizedBox(plate.x1, plate.y1, plate.x2, plate.y2),
        plate_text=observation.text.text,
        attributes=attributes,
        evidence=evidence,
    )
