"""Deterministic person-down candidate evaluation against bounded labeled intervals.

This module measures candidate alerts; it does not infer falls, injury, cause,
fault, or intent. Matching is one-to-one and deterministic so repeated alerts
cannot inflate recall. Input events are treated as untrusted structured data.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Iterable

_MAX_TIMESTAMP_MS = 253402300799999
_MAX_EVENTS = 100_000
_MAX_LABELS = 100_000


def _timestamp(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_TIMESTAMP_MS:
        raise ValueError(f"{name} must be an integer timestamp in the supported UTC range")


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
        if not isinstance(self.label_id, str) or not self.label_id or len(self.label_id) > 128:
            raise ValueError("label_id must be a bounded nonempty string")


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


def evaluate_person_down_candidates(
    events: Iterable[dict],
    labels: Iterable[LabeledPersonDown],
    *,
    video_start_timestamp_ms: int,
    video_end_timestamp_ms: int,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """Measure one camera/video interval with one-to-one alert/label matching.

    A candidate matches a labeled episode when its evidence window overlaps the
    label after applying the configured time tolerance. Each candidate and each
    label can match at most once. Extra candidate alerts remain false alerts.
    """
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

    unmatched = set(range(len(label_list)))
    matched_labels: list[str] = []
    delays: list[int] = []
    matched_events = 0
    tolerance = cfg.match_tolerance_ms

    for event_start, event_end in event_windows:
        candidates: list[tuple[int, int, str, int]] = []
        for index in unmatched:
            label = label_list[index]
            if event_end < label.start_timestamp_ms - tolerance:
                continue
            if event_start > label.end_timestamp_ms + tolerance:
                continue
            delay = event_end - label.start_timestamp_ms
            candidates.append((abs(delay), label.start_timestamp_ms, label.label_id, index))
        if not candidates:
            continue
        _, _, _, match_index = min(candidates)
        label = label_list[match_index]
        unmatched.remove(match_index)
        matched_events += 1
        matched_labels.append(label.label_id)
        delays.append(event_end - label.start_timestamp_ms)

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
    )
