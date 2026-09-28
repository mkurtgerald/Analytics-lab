"""Operator-safe forensic result projections and playback references.

These projections contain normalized analytic metadata and evidence provenance
only. A PlaybackReference is a non-authorizing pointer that a consuming VMS may
resolve under its own access-control and evidence policies. Nothing here reads,
stores, exports, or authorizes media.
"""
from __future__ import annotations

from dataclasses import dataclass

from .forensic_search import (
    AttributeFilter,
    ForensicAttribute,
    ForensicEvidenceLink,
    ForensicRecord,
)
from .forensic_temporal import AttributeCandidateTrail, TemporalTrail

_MAX_ROLL_MS = 300_000


def _roll(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_ROLL_MS:
        raise ValueError(f"{name} must be an integer in [0, 300000]")
    return value


@dataclass(frozen=True)
class PlaybackReference:
    """Non-authorizing pointer to evidence-relative playback context."""

    source_id: str
    timestamp_ms: int
    evidence: ForensicEvidenceLink
    pre_roll_ms: int = 5_000
    post_roll_ms: int = 5_000
    authorizes_action: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("playback source_id must be nonempty")
        if type(self.timestamp_ms) is not int or self.timestamp_ms < 0:
            raise ValueError("playback timestamp_ms must be nonnegative")
        if not isinstance(self.evidence, ForensicEvidenceLink):
            raise ValueError("playback reference requires immutable evidence")
        object.__setattr__(self, "pre_roll_ms", _roll(self.pre_roll_ms, "pre_roll_ms"))
        object.__setattr__(self, "post_roll_ms", _roll(self.post_roll_ms, "post_roll_ms"))
        if self.authorizes_action is not False:
            raise ValueError("playback reference must remain non-authorizing")


@dataclass(frozen=True)
class ForensicOperatorHit:
    """One evidence-linked forensic hit suitable for operator presentation."""

    observation_id: str
    source_id: str
    timestamp_ms: int
    category: str
    confidence: float
    evidence: ForensicEvidenceLink
    playback: PlaybackReference
    plate_text: str | None = None
    local_track_id: str | None = None
    attributes: tuple[ForensicAttribute, ...] = ()
    identity_claim: bool = False
    authorizes_action: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, ForensicEvidenceLink):
            raise ValueError("operator hit requires immutable evidence")
        if not isinstance(self.playback, PlaybackReference):
            raise ValueError("operator hit requires PlaybackReference")
        if (
            self.playback.source_id != self.source_id
            or self.playback.timestamp_ms != self.timestamp_ms
            or self.playback.evidence != self.evidence
        ):
            raise ValueError("playback reference must bind the same hit/evidence")
        if type(self.attributes) is not tuple or any(
            not isinstance(item, ForensicAttribute) for item in self.attributes
        ):
            raise ValueError("operator hit attributes are invalid")
        if self.identity_claim is not False:
            raise ValueError("forensic operator hit must not claim identity")
        if self.authorizes_action is not False:
            raise ValueError("forensic operator hit must remain non-authorizing")


@dataclass(frozen=True)
class ForensicTrailResult:
    """Operator projection of one cross-camera temporal/candidate trail."""

    association_kind: str
    hits: tuple[ForensicOperatorHit, ...]
    association_value: str | None = None
    required_attributes: tuple[AttributeFilter, ...] = ()
    identity_claim: bool = False
    authorizes_action: bool = False

    def __post_init__(self) -> None:
        if self.association_kind not in {"exact_plate_text", "exact_attribute_candidate"}:
            raise ValueError("unsupported forensic trail result kind")
        if type(self.hits) is not tuple or len(self.hits) < 2:
            raise ValueError("forensic trail result requires at least two hits")
        if len({hit.source_id for hit in self.hits}) < 2:
            raise ValueError("forensic trail result requires distinct sources")
        if any(not isinstance(hit, ForensicOperatorHit) for hit in self.hits):
            raise ValueError("forensic trail result contains an invalid hit")
        if self.identity_claim is not False:
            raise ValueError("forensic trail result must not claim identity")
        if self.authorizes_action is not False:
            raise ValueError("forensic trail result must remain non-authorizing")

        if self.association_kind == "exact_plate_text":
            if not isinstance(self.association_value, str) or not self.association_value:
                raise ValueError("exact plate result requires association_value")
            if self.required_attributes:
                raise ValueError("exact plate result cannot carry attribute filters")
            if any(hit.plate_text != self.association_value for hit in self.hits):
                raise ValueError("plate result contains a nonmatching hit")
        else:
            if self.association_value is not None:
                raise ValueError("attribute result does not use association_value")
            if type(self.required_attributes) is not tuple or not self.required_attributes:
                raise ValueError("attribute result requires exact attribute filters")
            canonical = tuple(
                AttributeFilter(name, value)
                for name, value in sorted(
                    {(item.name, item.value) for item in self.required_attributes}
                )
            )
            if self.required_attributes != canonical:
                raise ValueError("attribute filters must be canonical and deduplicated")


def operator_hit_from_record(
    record: ForensicRecord,
    *,
    pre_roll_ms: int = 5_000,
    post_roll_ms: int = 5_000,
) -> ForensicOperatorHit:
    if not isinstance(record, ForensicRecord):
        raise ValueError("record must be ForensicRecord")
    if record.evidence is None:
        raise ValueError("operator hit requires record evidence")
    playback = PlaybackReference(
        source_id=record.source_id,
        timestamp_ms=record.timestamp_ms,
        evidence=record.evidence,
        pre_roll_ms=pre_roll_ms,
        post_roll_ms=post_roll_ms,
    )
    return ForensicOperatorHit(
        observation_id=record.observation_id,
        source_id=record.source_id,
        timestamp_ms=record.timestamp_ms,
        category=record.category,
        confidence=record.confidence,
        evidence=record.evidence,
        playback=playback,
        plate_text=record.plate_text,
        local_track_id=record.track_id,
        attributes=record.attributes,
    )


def operator_result_from_plate_trail(
    trail: TemporalTrail,
    *,
    pre_roll_ms: int = 5_000,
    post_roll_ms: int = 5_000,
) -> ForensicTrailResult:
    if not isinstance(trail, TemporalTrail):
        raise ValueError("trail must be TemporalTrail")
    hits = tuple(
        operator_hit_from_record(
            record,
            pre_roll_ms=pre_roll_ms,
            post_roll_ms=post_roll_ms,
        )
        for record in trail.records
    )
    return ForensicTrailResult(
        association_kind="exact_plate_text",
        association_value=trail.association_value,
        hits=hits,
    )


def operator_result_from_attribute_trail(
    trail: AttributeCandidateTrail,
    *,
    pre_roll_ms: int = 5_000,
    post_roll_ms: int = 5_000,
) -> ForensicTrailResult:
    if not isinstance(trail, AttributeCandidateTrail):
        raise ValueError("trail must be AttributeCandidateTrail")
    hits = tuple(
        operator_hit_from_record(
            record,
            pre_roll_ms=pre_roll_ms,
            post_roll_ms=post_roll_ms,
        )
        for record in trail.records
    )
    return ForensicTrailResult(
        association_kind="exact_attribute_candidate",
        required_attributes=trail.required_attributes,
        hits=hits,
    )
