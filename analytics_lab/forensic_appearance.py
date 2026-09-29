"""Model-free non-biometric appearance similarity ranking.

This first appearance-search gate uses normalized color-histogram descriptors
only. It introduces no learned weights, ReID, face recognition, biometric
identity, raw-media storage, or vector-database dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import math
from typing import Iterable

from .forensic_search import ForensicRecord

_SCHEMA = "analytics.appearance-descriptor.v1"
_KIND = "model_free.color_histogram.v1"
_MAX_DIMS = 512
_MAX_CANDIDATES = 10_000


def _finite_nonnegative(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class AppearanceDescriptor:
    schema: str
    kind: str
    record: ForensicRecord
    values: tuple[float, ...]
    identity_claim: bool = False
    authorizes_action: bool = False

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("unsupported appearance descriptor schema")
        if self.kind != _KIND:
            raise ValueError("unsupported appearance descriptor kind")
        if not isinstance(self.record, ForensicRecord):
            raise ValueError("record must be ForensicRecord")
        if self.record.evidence is None:
            raise ValueError("appearance descriptor requires immutable evidence")
        if self.record.category in {"face", "license_plate"}:
            raise ValueError("appearance descriptor category is not permitted")
        if type(self.values) is not tuple or not 2 <= len(self.values) <= _MAX_DIMS:
            raise ValueError("appearance descriptor dimension is outside supported bound")
        normalized = tuple(_finite_nonnegative(v, "descriptor value") for v in self.values)
        total = sum(normalized)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("appearance descriptor must be L1 normalized")
        object.__setattr__(self, "values", normalized)
        if self.identity_claim is not False:
            raise ValueError("appearance descriptor must not claim identity")
        if self.authorizes_action is not False:
            raise ValueError("appearance descriptor must remain non-authorizing")


@dataclass(frozen=True)
class AppearanceMatch:
    probe_observation_id: str
    candidate: ForensicRecord
    similarity: float
    identity_claim: bool = False
    authorizes_action: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ForensicRecord):
            raise ValueError("candidate must be ForensicRecord")
        score = _finite_nonnegative(self.similarity, "similarity")
        if score > 1.0:
            raise ValueError("similarity must be within [0, 1]")
        object.__setattr__(self, "similarity", score)
        if self.identity_claim is not False:
            raise ValueError("appearance match must not claim identity")
        if self.authorizes_action is not False:
            raise ValueError("appearance match must remain non-authorizing")


def histogram_intersection(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("descriptor dimensions must match")
    return float(sum(min(x, y) for x, y in zip(a, b)))


def rank_appearance_similarity(
    probe: AppearanceDescriptor,
    candidates: Iterable[AppearanceDescriptor],
    *,
    min_similarity: float = 0.0,
    limit: int = 100,
) -> tuple[AppearanceMatch, ...]:
    if not isinstance(probe, AppearanceDescriptor):
        raise ValueError("probe must be AppearanceDescriptor")
    threshold = _finite_nonnegative(min_similarity, "min_similarity")
    if threshold > 1.0:
        raise ValueError("min_similarity must be within [0, 1]")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer in [1, 1000]")
    try:
        iterator = iter(candidates)
    except TypeError as exc:
        raise ValueError("candidates must be iterable") from exc
    items = tuple(islice(iterator, _MAX_CANDIDATES + 1))
    if len(items) > _MAX_CANDIDATES:
        raise RuntimeError("appearance candidate count exceeds supported bound")
    if any(not isinstance(item, AppearanceDescriptor) for item in items):
        raise ValueError("candidates must contain AppearanceDescriptor values")

    ids = [item.record.observation_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate appearance candidate observation_id")

    matches: list[AppearanceMatch] = []
    for item in items:
        if item.record.observation_id == probe.record.observation_id:
            continue
        if item.schema != probe.schema or item.kind != probe.kind:
            raise ValueError("appearance descriptor kind/schema mismatch")
        if item.record.category != probe.record.category:
            raise ValueError("appearance descriptor category mismatch")
        if len(item.values) != len(probe.values):
            raise ValueError("appearance descriptor dimension mismatch")
        score = histogram_intersection(probe.values, item.values)
        if score < threshold and not math.isclose(score, threshold, rel_tol=0.0, abs_tol=1e-9):
            continue
        matches.append(
            AppearanceMatch(
                probe_observation_id=probe.record.observation_id,
                candidate=item.record,
                similarity=score,
            )
        )

    matches.sort(
        key=lambda item: (
            -item.similarity,
            item.candidate.timestamp_ms,
            item.candidate.source_id,
            item.candidate.observation_id,
        )
    )
    return tuple(matches[:limit])
