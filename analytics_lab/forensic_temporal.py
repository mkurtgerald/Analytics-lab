"""Bounded cross-camera temporal association for forensic search.

This module creates evidence-linked temporal trails from already-searchable
forensic metadata. It does not assert person/vehicle identity. The first gate
supports exact normalized plate-text association only.

A prefix query may narrow retrieval, but records are grouped strictly by each
record's exact normalized plate text so different plates that share a prefix are
never cross-linked merely because of that prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .forensic_search import ForensicQuery, ForensicRecord


class ForensicSearcher(Protocol):
    def search(self, query: ForensicQuery) -> tuple[ForensicRecord, ...]: ...


@dataclass(frozen=True)
class TemporalAssociationConfig:
    max_link_gap_ms: int = 60_000
    max_total_span_ms: int = 300_000
    max_records_per_trail: int = 32
    max_sources_per_trail: int = 8

    def __post_init__(self) -> None:
        for name, upper in (
            ("max_link_gap_ms", 86_400_000),
            ("max_total_span_ms", 86_400_000),
            ("max_records_per_trail", 256),
            ("max_sources_per_trail", 64),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} is outside the supported bound")
        if self.max_total_span_ms < self.max_link_gap_ms:
            raise ValueError("max_total_span_ms must be >= max_link_gap_ms")


@dataclass(frozen=True)
class TemporalTrail:
    association_kind: str
    association_value: str
    records: tuple[ForensicRecord, ...]
    identity_claim: bool = False

    def __post_init__(self) -> None:
        if self.association_kind != "exact_plate_text":
            raise ValueError("unsupported temporal association kind")
        if not isinstance(self.association_value, str) or not self.association_value:
            raise ValueError("association_value must be nonempty")
        if type(self.records) is not tuple or len(self.records) < 2:
            raise ValueError("temporal trail requires at least two records")
        if self.identity_claim is not False:
            raise ValueError("temporal trails must never claim identity")
        ordered = tuple(
            sorted(
                self.records,
                key=lambda item: (item.timestamp_ms, item.source_id, item.observation_id),
            )
        )
        if ordered != self.records:
            raise ValueError("temporal trail records must be deterministically ordered")
        if any(
            item.category != "license_plate" or item.plate_text != self.association_value
            for item in self.records
        ):
            raise ValueError("temporal trail contains a nonmatching plate record")
        if len({item.source_id for item in self.records}) < 2:
            raise ValueError("cross-camera temporal trail requires distinct sources")
        if any(item.evidence is None for item in self.records):
            raise ValueError("temporal trail records require immutable evidence provenance")


def _emit_if_cross_source(
    records: list[ForensicRecord],
    *,
    plate_text: str,
) -> TemporalTrail | None:
    if len(records) < 2 or len({item.source_id for item in records}) < 2:
        return None
    return TemporalTrail(
        association_kind="exact_plate_text",
        association_value=plate_text,
        records=tuple(records),
    )


def exact_plate_temporal_trails(
    searcher: ForensicSearcher,
    query: ForensicQuery,
    *,
    config: TemporalAssociationConfig | None = None,
) -> tuple[TemporalTrail, ...]:
    """Build bounded cross-camera trails using exact plate text only.

    Query filters are applied by the supplied forensic search backend. Prefix
    plate queries remain retrieval filters only: each returned record is grouped
    by its own exact plate text before temporal association.
    """
    if not callable(getattr(searcher, "search", None)):
        raise ValueError("searcher must expose search")
    if not isinstance(query, ForensicQuery):
        raise ValueError("query must be ForensicQuery")
    cfg = config or TemporalAssociationConfig()
    if not isinstance(cfg, TemporalAssociationConfig):
        raise ValueError("config must be TemporalAssociationConfig")

    records = searcher.search(query)
    if not isinstance(records, tuple):
        raise ValueError("searcher must return a tuple of ForensicRecord values")
    if len(records) > query.limit:
        raise ValueError("searcher returned more records than the query limit")
    if any(not isinstance(item, ForensicRecord) for item in records):
        raise ValueError("searcher returned an unsupported record")

    groups: dict[str, list[ForensicRecord]] = {}
    for record in records:
        if record.category != "license_plate" or record.plate_text is None:
            continue
        groups.setdefault(record.plate_text, []).append(record)

    trails: list[TemporalTrail] = []
    for plate_text in sorted(groups):
        ordered = sorted(
            groups[plate_text],
            key=lambda item: (item.timestamp_ms, item.source_id, item.observation_id),
        )
        segment: list[ForensicRecord] = []
        sources: set[str] = set()

        for record in ordered:
            if not segment:
                segment = [record]
                sources = {record.source_id}
                continue

            previous = segment[-1]
            new_source_count = len(sources | {record.source_id})
            exceeds = (
                record.timestamp_ms - previous.timestamp_ms > cfg.max_link_gap_ms
                or record.timestamp_ms - segment[0].timestamp_ms > cfg.max_total_span_ms
                or len(segment) >= cfg.max_records_per_trail
                or new_source_count > cfg.max_sources_per_trail
            )
            if exceeds:
                trail = _emit_if_cross_source(segment, plate_text=plate_text)
                if trail is not None:
                    trails.append(trail)
                segment = [record]
                sources = {record.source_id}
            else:
                segment.append(record)
                sources.add(record.source_id)

        trail = _emit_if_cross_source(segment, plate_text=plate_text)
        if trail is not None:
            trails.append(trail)

    trails.sort(
        key=lambda item: (
            item.records[0].timestamp_ms,
            item.association_value,
            item.records[0].source_id,
            item.records[0].observation_id,
        )
    )
    return tuple(trails)
