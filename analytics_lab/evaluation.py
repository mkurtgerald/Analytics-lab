"""Deterministic person-down candidate evaluation against bounded labeled intervals.

This module measures candidate alerts; it does not infer falls, injury, cause,
fault, or intent. Matching is one-to-one and deterministic so repeated alerts
cannot inflate recall. Input events are treated as untrusted structured data.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from statistics import median
from typing import Iterable

_MAX_TIMESTAMP_MS = 253402300799999
_MAX_EVENTS = 100_000
_MAX_LABELS = 100_000
_MAX_SAMPLES = 100_000
_MAX_ID_LENGTH = 128


def _timestamp(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_TIMESTAMP_MS:
        raise ValueError(f"{name} must be an integer timestamp in the supported UTC range")


def _bounded_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_ID_LENGTH:
        raise ValueError(f"{name} must be a bounded nonempty string")


@dataclass(frozen=True, order=True)
class LabeledPersonDown:
    start_timestamp_ms: int
    end_timestamp_ms: int
    label_id: str

    def __post_init__(self) -> None:
        _timestamp(self.start_timestamp_ms, "start_timestamp_ms")
        _timestamp(self.end_timestamp_ms, "end_timestamp_ms")
        if self.end_timestamp_ms < self.start_timestamp_ms:
            raise ValueError("label end must not precede label start")
        _bounded_id(self.label_id, "label_id")


@dataclass(frozen=True)
class EvaluationConfig:
    match_tolerance_ms: int = 1000

    def __post_init__(self) -> None:
        if type(self.match_tolerance_ms) is not int or self.match_tolerance_ms < 0:
            raise ValueError("match_tolerance_ms must be an integer >= 0")


@dataclass(frozen=True)
class EvaluationResult:
    duration_ms: int
    positive_episodes: int
    candidate_events: int
    matched_events: int
    missed_episodes: int
    false_alerts: int
    precision: float | None
    recall: float | None
    false_alerts_per_camera_hour: float
    median_alert_delay_ms: float | None
    matched_label_ids: tuple[str, ...]
    alert_delays_ms: tuple[int, ...] = ()


@dataclass(frozen=True)
class EvaluationSample:
    """One independently evaluated, time-bounded camera sample."""

    sample_id: str
    site_id: str
    camera_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    result: EvaluationResult

    def __post_init__(self) -> None:
        _bounded_id(self.sample_id, "sample_id")
        _bounded_id(self.site_id, "site_id")
        _bounded_id(self.camera_id, "camera_id")
        _timestamp(self.start_timestamp_ms, "start_timestamp_ms")
        _timestamp(self.end_timestamp_ms, "end_timestamp_ms")
        if self.end_timestamp_ms <= self.start_timestamp_ms:
            raise ValueError("sample interval must have positive duration")
        if not isinstance(self.result, EvaluationResult):
            raise ValueError("result must be an EvaluationResult")
        if self.result.duration_ms != self.end_timestamp_ms - self.start_timestamp_ms:
            raise ValueError("sample interval must match evaluation duration")


@dataclass(frozen=True)
class EvaluationGroupResult:
    group_id: str
    sample_count: int
    duration_ms: int
    camera_hours: float
    positive_episodes: int
    candidate_events: int
    matched_events: int
    missed_episodes: int
    false_alerts: int
    precision: float | None
    recall: float | None
    false_alerts_per_camera_hour: float
    median_alert_delay_ms: float | None
    site_id: str | None = None
    camera_id: str | None = None


@dataclass(frozen=True)
class EvaluationAggregate:
    sample_count: int
    site_count: int
    camera_count: int
    duration_ms: int
    camera_hours: float
    positive_episodes: int
    candidate_events: int
    matched_events: int
    missed_episodes: int
    false_alerts: int
    precision: float | None
    recall: float | None
    false_alerts_per_camera_hour: float
    median_alert_delay_ms: float | None
    by_site: tuple[EvaluationGroupResult, ...]
    by_camera: tuple[EvaluationGroupResult, ...]


def _event_window(event: object) -> tuple[int, int]:
    if not isinstance(event, dict):
        raise ValueError("candidate event must be an object")
    if event.get("event_type") != "person_down_candidate" or event.get("status") != "candidate":
        raise ValueError("unsupported event type or status")
    evidence = event.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1 or not isinstance(evidence[0], dict):
        raise ValueError("candidate event must contain exactly one evidence window")
    window = evidence[0]
    if window.get("type") != "observation_window":
        raise ValueError("candidate event evidence must be an observation window")
    start = window.get("start_timestamp_ms")
    end = window.get("end_timestamp_ms")
    _timestamp(start, "event start_timestamp_ms")
    _timestamp(end, "event end_timestamp_ms")
    if end < start:
        raise ValueError("event evidence end must not precede start")
    return start, end


def _maximum_cardinality_matches(
    event_windows: list[tuple[int, int]],
    labels: list[LabeledPersonDown],
    tolerance_ms: int,
) -> list[tuple[int, int]]:
    """Return a deterministic maximum-cardinality interval matching.

    Events are processed by increasing end time. Among labels whose
    tolerance-expanded interval overlaps the current event, matching the label
    with the earliest expanded end is cardinality-safe: any maximum matching
    that used a later-ending compatible label can exchange the two labels
    without making the later event incompatible. This avoids the quadratic
    candidate scan and, unlike closest-delay greedy matching, cannot consume a
    label needed by a later event and under-count recall.
    """
    ordered_labels = sorted(
        enumerate(labels),
        key=lambda item: (
            item[1].start_timestamp_ms - tolerance_ms,
            item[1].end_timestamp_ms + tolerance_ms,
            item[1].label_id,
        ),
    )
    active: list[tuple[int, int, str, int]] = []
    next_label = 0
    matches: list[tuple[int, int]] = []

    for event_index, (event_start, event_end) in enumerate(event_windows):
        while next_label < len(ordered_labels):
            label_index, label = ordered_labels[next_label]
            expanded_start = label.start_timestamp_ms - tolerance_ms
            if expanded_start > event_end:
                break
            heapq.heappush(
                active,
                (
                    label.end_timestamp_ms + tolerance_ms,
                    label.start_timestamp_ms,
                    label.label_id,
                    label_index,
                ),
            )
            next_label += 1
        while active and active[0][0] < event_start:
            heapq.heappop(active)
        if active:
            _, _, _, label_index = heapq.heappop(active)
            matches.append((event_index, label_index))
    return matches


def evaluate_person_down_candidates(
    events: Iterable[dict],
    labels: Iterable[LabeledPersonDown],
    *,
    video_start_timestamp_ms: int,
    video_end_timestamp_ms: int,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """Measure one camera/video interval with one-to-one alert/label matching."""
    _timestamp(video_start_timestamp_ms, "video_start_timestamp_ms")
    _timestamp(video_end_timestamp_ms, "video_end_timestamp_ms")
    if video_end_timestamp_ms <= video_start_timestamp_ms:
        raise ValueError("video interval must have positive duration")
    cfg = config or EvaluationConfig()
    if not isinstance(cfg, EvaluationConfig):
        raise ValueError("config must be an EvaluationConfig")

    label_list = list(labels)
    if len(label_list) > _MAX_LABELS:
        raise RuntimeError("label limit exceeded")
    if any(not isinstance(item, LabeledPersonDown) for item in label_list):
        raise ValueError("labels must contain LabeledPersonDown values")
    label_ids = [item.label_id for item in label_list]
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("label_id values must be unique")
    for label in label_list:
        if label.start_timestamp_ms < video_start_timestamp_ms or label.end_timestamp_ms > video_end_timestamp_ms:
            raise ValueError("label lies outside the evaluated video interval")
    label_list.sort(key=lambda item: (item.start_timestamp_ms, item.end_timestamp_ms, item.label_id))

    event_windows: list[tuple[int, int]] = []
    for event in events:
        if len(event_windows) >= _MAX_EVENTS:
            raise RuntimeError("candidate event limit exceeded")
        start, end = _event_window(event)
        if start < video_start_timestamp_ms or end > video_end_timestamp_ms:
            raise ValueError("candidate event lies outside the evaluated video interval")
        event_windows.append((start, end))
    event_windows.sort(key=lambda item: (item[1], item[0]))

    matches = _maximum_cardinality_matches(event_windows, label_list, cfg.match_tolerance_ms)
    matched_labels: list[str] = []
    delays: list[int] = []
    for event_index, label_index in matches:
        _, event_end = event_windows[event_index]
        label = label_list[label_index]
        matched_labels.append(label.label_id)
        delays.append(event_end - label.start_timestamp_ms)

    matched_events = len(matches)
    positive_count = len(label_list)
    event_count = len(event_windows)
    false_alerts = event_count - matched_events
    missed = positive_count - matched_events
    duration_ms = video_end_timestamp_ms - video_start_timestamp_ms
    duration_hours = duration_ms / 3_600_000.0
    precision = matched_events / event_count if event_count else None
    recall = matched_events / positive_count if positive_count else None
    false_rate = false_alerts / duration_hours
    delay_median = float(median(delays)) if delays else None
    for value in (precision, recall, false_rate, delay_median):
        if value is not None and not math.isfinite(value):
            raise RuntimeError("non-finite evaluation metric")
    return EvaluationResult(
        duration_ms=duration_ms,
        positive_episodes=positive_count,
        candidate_events=event_count,
        matched_events=matched_events,
        missed_episodes=missed,
        false_alerts=false_alerts,
        precision=precision,
        recall=recall,
        false_alerts_per_camera_hour=false_rate,
        median_alert_delay_ms=delay_median,
        matched_label_ids=tuple(matched_labels),
        alert_delays_ms=tuple(delays),
    )


def _summarize(
    group_id: str,
    samples: list[EvaluationSample],
    *,
    site_id: str | None = None,
    camera_id: str | None = None,
) -> EvaluationGroupResult:
    _bounded_id(group_id, "group_id")
    if site_id is not None:
        _bounded_id(site_id, "site_id")
    if camera_id is not None:
        _bounded_id(camera_id, "camera_id")
    duration_ms = sum(item.result.duration_ms for item in samples)
    positive = sum(item.result.positive_episodes for item in samples)
    events = sum(item.result.candidate_events for item in samples)
    matched = sum(item.result.matched_events for item in samples)
    missed = sum(item.result.missed_episodes for item in samples)
    false_alerts = sum(item.result.false_alerts for item in samples)
    delays = [delay for item in samples for delay in item.result.alert_delays_ms]
    camera_hours = duration_ms / 3_600_000.0
    precision = matched / events if events else None
    recall = matched / positive if positive else None
    false_rate = false_alerts / camera_hours
    delay_median = float(median(delays)) if delays else None
    return EvaluationGroupResult(
        group_id=group_id,
        sample_count=len(samples),
        duration_ms=duration_ms,
        camera_hours=camera_hours,
        positive_episodes=positive,
        candidate_events=events,
        matched_events=matched,
        missed_episodes=missed,
        false_alerts=false_alerts,
        precision=precision,
        recall=recall,
        false_alerts_per_camera_hour=false_rate,
        median_alert_delay_ms=delay_median,
        site_id=site_id,
        camera_id=camera_id,
    )


def aggregate_person_down_evaluations(samples: Iterable[EvaluationSample]) -> EvaluationAggregate:
    """Aggregate disjoint evaluated intervals across cameras and sites.

    Overlapping intervals for the same site-scoped camera are rejected so
    repeated or partially duplicated footage cannot silently inflate
    camera-hours or event counts. Aggregate precision/recall are micro-averaged
    from raw counts.
    """
    items = list(samples)
    if not items:
        raise ValueError("at least one evaluation sample is required")
    if len(items) > _MAX_SAMPLES:
        raise RuntimeError("evaluation sample limit exceeded")
    if any(not isinstance(item, EvaluationSample) for item in items):
        raise ValueError("samples must contain EvaluationSample values")
    ids = [item.sample_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("sample_id values must be unique")

    camera_intervals: dict[tuple[str, str], list[EvaluationSample]] = {}
    site_groups: dict[str, list[EvaluationSample]] = {}
    for item in items:
        camera_intervals.setdefault((item.site_id, item.camera_id), []).append(item)
        site_groups.setdefault(item.site_id, []).append(item)
    for group in camera_intervals.values():
        ordered = sorted(group, key=lambda item: (item.start_timestamp_ms, item.end_timestamp_ms, item.sample_id))
        previous_end = None
        for item in ordered:
            if previous_end is not None and item.start_timestamp_ms < previous_end:
                raise ValueError("overlapping evaluation intervals for the same site/camera")
            previous_end = item.end_timestamp_ms

    total = _summarize("all", items)
    by_site = tuple(_summarize(key, site_groups[key], site_id=key) for key in sorted(site_groups))
    by_camera = tuple(
        _summarize(
            camera_id,
            camera_intervals[(site_id, camera_id)],
            site_id=site_id,
            camera_id=camera_id,
        )
        for site_id, camera_id in sorted(camera_intervals)
    )
    return EvaluationAggregate(
        sample_count=total.sample_count,
        site_count=len(site_groups),
        camera_count=len(camera_intervals),
        duration_ms=total.duration_ms,
        camera_hours=total.camera_hours,
        positive_episodes=total.positive_episodes,
        candidate_events=total.candidate_events,
        matched_events=total.matched_events,
        missed_episodes=total.missed_episodes,
        false_alerts=total.false_alerts,
        precision=total.precision,
        recall=total.recall,
        false_alerts_per_camera_hour=total.false_alerts_per_camera_hour,
        median_alert_delay_ms=total.median_alert_delay_ms,
        by_site=by_site,
        by_camera=by_camera,
    )
