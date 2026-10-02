"""Versioned serialized forensic-search handoff for K5/VMS consumers.

The handoff is metadata-only and non-authorizing. It serializes operator-safe
forensic results into a deterministic JSON-compatible envelope without raw
media, credentials, source URIs, biometric templates, or identity claims.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from .forensic_appearance import AppearanceMatch
from .forensic_results import (
    ForensicOperatorHit, ForensicTrailResult, PlaybackReference, operator_hit_from_record,
)
from .forensic_search import (
    AttributeFilter, ForensicAttribute, ForensicEvidenceLink, ForensicRecord, _token,
)
from .forensic_temporal import AttributeCandidateTrail, TemporalTrail

_SCHEMA = "analytics.forensic-result.v1"
_MAX_JSON_BYTES = 1_000_000
_FLAGS = {"identity_claim", "authorizes_action"}
_HIT_FIELDS = {
    "observation_id", "source_id", "timestamp_ms", "category", "confidence",
    "plate_text", "local_track_id", "attributes", "evidence", "playback", *_FLAGS,
}


def _fields(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has missing or unexpected fields")
    return value


def _non_authorizing(value: dict) -> None:
    if value.get("identity_claim") is not False:
        raise ValueError("serialized forensic result must not claim identity")
    if value.get("authorizes_action") is not False:
        raise ValueError("serialized forensic result must remain non-authorizing")


def _evidence(value: object) -> ForensicEvidenceLink:
    fields = _fields(value, {"event_id", "producer", "producer_version", "config_sha256",
                             "model_sha256", "source_revision"}, "evidence")
    return ForensicEvidenceLink(**fields)


def _hit(value: object) -> tuple[ForensicOperatorHit, ForensicRecord]:
    fields = _fields(value, _HIT_FIELDS, "hit")
    _non_authorizing(fields)
    attributes = fields["attributes"]
    if not isinstance(attributes, list) or len(attributes) > 64:
        raise ValueError("hit attributes must be a bounded list")
    attributes = tuple(ForensicAttribute(**_fields(
        item, {"name", "value", "confidence", "provenance"}, "attribute")) for item in attributes)
    evidence = _evidence(fields["evidence"])
    record = ForensicRecord(
        observation_id=fields["observation_id"], source_id=fields["source_id"],
        timestamp_ms=fields["timestamp_ms"], category=fields["category"],
        confidence=fields["confidence"], plate_text=fields["plate_text"],
        track_id=fields["local_track_id"], attributes=attributes, evidence=evidence,
    )
    playback = _fields(fields["playback"], {"source_id", "timestamp_ms", "pre_roll_ms",
                                           "post_roll_ms", "evidence", "authorizes_action"}, "playback")
    playback = PlaybackReference(**{**playback, "evidence": _evidence(playback["evidence"])})
    hit = operator_hit_from_record(record, pre_roll_ms=playback.pre_roll_ms,
                                   post_roll_ms=playback.post_roll_ms)
    if hit.playback != playback:
        raise ValueError("playback reference must bind the same hit/evidence")
    if _hit_payload(hit) != fields:
        raise ValueError("hit must contain canonical normalized metadata")
    return hit, record


def _validate_payload(payload: dict) -> None:
    _non_authorizing(payload)
    kind = payload.get("kind")
    if kind == "hit":
        _fields(payload, {"kind", *_HIT_FIELDS}, "hit payload")
        _hit({key: value for key, value in payload.items() if key != "kind"})
    elif kind == "trail":
        _fields(payload, {"kind", "association_kind", "association_value", "required_attributes",
                          "hits", *_FLAGS}, "trail payload")
        if not isinstance(payload["hits"], list):
            raise ValueError("trail hits must be a list")
        if not isinstance(payload["required_attributes"], list):
            raise ValueError("trail attributes must be a list")
        filters = tuple(AttributeFilter(**_fields(item, {"name", "value"}, "attribute filter"))
                        for item in payload["required_attributes"])
        hits_and_records = tuple(_hit(item) for item in payload["hits"])
        ForensicTrailResult(
            association_kind=payload["association_kind"], association_value=payload["association_value"],
            hits=tuple(item[0] for item in hits_and_records), required_attributes=filters,
        )
        records = tuple(item[1] for item in hits_and_records)
        if payload["association_kind"] == "exact_plate_text":
            TemporalTrail("exact_plate_text", payload["association_value"], records)
        else:
            AttributeCandidateTrail(records[0].category, filters, records)
        if [asdict(item) for item in filters] != payload["required_attributes"]:
            raise ValueError("trail attributes must be canonical")
    elif kind == "appearance_matches":
        _fields(payload, {"kind", "probe_observation_id", "descriptor_schema", "descriptor_kind",
                          "matches", *_FLAGS}, "appearance payload")
        _token(payload["probe_observation_id"], "probe_observation_id")
        if (payload["descriptor_schema"] != "analytics.appearance-descriptor.v1"
                or payload["descriptor_kind"] != "model_free.color_histogram.v1"):
            raise ValueError("unsupported appearance descriptor schema/kind")
        values = payload["matches"]
        if not isinstance(values, list) or not 1 <= len(values) <= 1000:
            raise ValueError("appearance handoff requires a bounded nonempty match list")
        matches = []
        for value in values:
            _fields(value, {"similarity", "hit", *_FLAGS}, "appearance match")
            _non_authorizing(value)
            matches.append(AppearanceMatch(payload["probe_observation_id"], _hit(value["hit"])[1],
                                           value["similarity"]))
        _validate_matches(tuple(matches))
    else:
        raise ValueError("unsupported forensic result kind")


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("serialized forensic result contains duplicate fields")
        result[key] = value
    return result


@dataclass(frozen=True)
class ForensicResultEnvelope:
    schema: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("unsupported forensic result schema")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dictionary")
        try:
            _validate_payload(self.payload)
        except (TypeError, OverflowError) as exc:
            raise ValueError("serialized forensic result has invalid field values") from exc

    def to_json(self) -> str:
        # A frozen dataclass does not freeze its nested dictionaries/lists.
        self.__post_init__()
        raw = json.dumps(
            {"schema": self.schema, "payload": self.payload},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(raw.encode("utf-8")) > _MAX_JSON_BYTES:
            raise ValueError("serialized forensic result is too large")
        return raw


def _hit_payload(hit: ForensicOperatorHit) -> dict[str, Any]:
    if not isinstance(hit, ForensicOperatorHit):
        raise ValueError("hit must be ForensicOperatorHit")
    return {
        "observation_id": hit.observation_id,
        "source_id": hit.source_id,
        "timestamp_ms": hit.timestamp_ms,
        "category": hit.category,
        "confidence": hit.confidence,
        "plate_text": hit.plate_text,
        "local_track_id": hit.local_track_id,
        "attributes": [asdict(item) for item in hit.attributes],
        "evidence": asdict(hit.evidence),
        "playback": {
            "source_id": hit.playback.source_id,
            "timestamp_ms": hit.playback.timestamp_ms,
            "pre_roll_ms": hit.playback.pre_roll_ms,
            "post_roll_ms": hit.playback.post_roll_ms,
            "evidence": asdict(hit.playback.evidence),
            "authorizes_action": False,
        },
        "identity_claim": False,
        "authorizes_action": False,
    }


def envelope_for_hit(hit: ForensicOperatorHit) -> ForensicResultEnvelope:
    payload = _hit_payload(hit)
    payload["kind"] = "hit"
    return ForensicResultEnvelope(schema=_SCHEMA, payload=payload)


def envelope_for_trail(result: ForensicTrailResult) -> ForensicResultEnvelope:
    if not isinstance(result, ForensicTrailResult):
        raise ValueError("result must be ForensicTrailResult")
    payload = {
        "kind": "trail",
        "association_kind": result.association_kind,
        "association_value": result.association_value,
        "required_attributes": [asdict(item) for item in result.required_attributes],
        "hits": [_hit_payload(hit) for hit in result.hits],
        "identity_claim": False,
        "authorizes_action": False,
    }
    return ForensicResultEnvelope(schema=_SCHEMA, payload=payload)


def _validate_matches(matches: tuple[AppearanceMatch, ...]) -> None:
    if type(matches) is not tuple or not matches:
        raise ValueError("appearance handoff requires a nonempty immutable match tuple")
    if len(matches) > 1000:
        raise ValueError("appearance handoff exceeds the supported result bound")
    if any(not isinstance(item, AppearanceMatch) for item in matches):
        raise ValueError("appearance handoff requires AppearanceMatch values")

    probe_ids = {item.probe_observation_id for item in matches}
    if len(probe_ids) != 1:
        raise ValueError("appearance handoff matches must share one probe observation")
    candidate_ids = [item.candidate.observation_id for item in matches]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("appearance handoff candidate observation IDs must be unique")

    ordered = tuple(
        sorted(
            matches,
            key=lambda item: (
                -item.similarity,
                item.candidate.timestamp_ms,
                item.candidate.source_id,
                item.candidate.observation_id,
            ),
        )
    )
    if ordered != matches:
        raise ValueError("appearance handoff matches must preserve deterministic ranking order")


def envelope_for_appearance_matches(
    matches: tuple[AppearanceMatch, ...],
    *,
    pre_roll_ms: int = 5_000,
    post_roll_ms: int = 5_000,
) -> ForensicResultEnvelope:
    """Serialize ordered non-biometric appearance matches for K5/VMS."""
    _validate_matches(matches)
    serialized_matches = []
    for item in matches:
        hit = operator_hit_from_record(
            item.candidate,
            pre_roll_ms=pre_roll_ms,
            post_roll_ms=post_roll_ms,
        )
        serialized_matches.append(
            {
                "similarity": item.similarity,
                "hit": _hit_payload(hit),
                "identity_claim": False,
                "authorizes_action": False,
            }
        )

    payload = {
        "kind": "appearance_matches",
        "probe_observation_id": matches[0].probe_observation_id,
        "descriptor_schema": "analytics.appearance-descriptor.v1",
        "descriptor_kind": "model_free.color_histogram.v1",
        "matches": serialized_matches,
        "identity_claim": False,
        "authorizes_action": False,
    }
    return ForensicResultEnvelope(schema=_SCHEMA, payload=payload)


def parse_envelope_json(raw: str) -> ForensicResultEnvelope:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError("serialized forensic result is invalid or too large")
    try:
        decoded = json.loads(raw, object_pairs_hook=_unique_fields)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("serialized forensic result is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("serialized forensic result must be an object")
    if set(decoded) != {"schema", "payload"}:
        raise ValueError("serialized forensic result has unexpected top-level fields")
    return ForensicResultEnvelope(
        schema=decoded["schema"],
        payload=decoded["payload"],
    )
