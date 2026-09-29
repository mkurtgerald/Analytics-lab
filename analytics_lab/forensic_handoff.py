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
from .forensic_results import ForensicOperatorHit, ForensicTrailResult, operator_hit_from_record

_SCHEMA = "analytics.forensic-result.v1"


@dataclass(frozen=True)
class ForensicResultEnvelope:
    schema: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("unsupported forensic result schema")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dictionary")
        if self.payload.get("identity_claim") is not False:
            raise ValueError("serialized forensic result must not claim identity")
        if self.payload.get("authorizes_action") is not False:
            raise ValueError("serialized forensic result must remain non-authorizing")

    def to_json(self) -> str:
        return json.dumps(
            {"schema": self.schema, "payload": self.payload},
            sort_keys=True,
            separators=(",", ":"),
        )


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


def envelope_for_appearance_matches(
    matches: tuple[AppearanceMatch, ...],
    *,
    pre_roll_ms: int = 5_000,
    post_roll_ms: int = 5_000,
) -> ForensicResultEnvelope:
    """Serialize ordered non-biometric appearance matches for K5/VMS."""
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
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 1_000_000:
        raise ValueError("serialized forensic result is invalid or too large")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("serialized forensic result is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("serialized forensic result must be an object")
    if set(decoded) != {"schema", "payload"}:
        raise ValueError("serialized forensic result has unexpected top-level fields")
    return ForensicResultEnvelope(
        schema=decoded["schema"],
        payload=decoded["payload"],
    )
