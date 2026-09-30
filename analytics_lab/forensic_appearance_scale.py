"""Bounded engineering scale probe for model-free appearance ranking.

This probe measures only in-memory descriptor scan/ranking time on deterministic
synthetic metadata descriptors. It uses no image/video pixels, learned weights,
ReID, biometric identity, or vector database. Timing is runner-specific
engineering evidence only and is not a performance SLA or release claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
import time
from typing import Callable

from .forensic_appearance import AppearanceDescriptor, rank_appearance_similarity
from .forensic_search import ForensicEvidenceLink, ForensicRecord


@dataclass(frozen=True)
class AppearanceScaleProbeConfig:
    candidate_count: int = 1000
    dimensions: int = 32
    iterations: int = 3
    result_limit: int = 100

    def __post_init__(self) -> None:
        if type(self.candidate_count) is not int or not 1 <= self.candidate_count <= 10_000:
            raise ValueError("candidate_count must be an integer in [1, 10000]")
        if type(self.dimensions) is not int or not 2 <= self.dimensions <= 512:
            raise ValueError("dimensions must be an integer in [2, 512]")
        if type(self.iterations) is not int or not 1 <= self.iterations <= 5:
            raise ValueError("iterations must be an integer in [1, 5]")
        if type(self.result_limit) is not int or not 1 <= self.result_limit <= 1000:
            raise ValueError("result_limit must be an integer in [1, 1000]")


@dataclass(frozen=True)
class AppearanceScaleProbeResult:
    candidate_count: int
    dimensions: int
    iterations: int
    result_limit: int
    result_count: int
    elapsed_ms: tuple[float, ...]
    median_ms: float
    max_ms: float
    identity_claim: bool = False
    authorizes_action: bool = False
    performance_claim: bool = False

    def __post_init__(self) -> None:
        if type(self.elapsed_ms) is not tuple or len(self.elapsed_ms) != self.iterations:
            raise ValueError("elapsed_ms must contain one timing per iteration")
        for value in self.elapsed_ms:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0.0:
                raise ValueError("elapsed_ms values must be finite and nonnegative")
        for name in ("median_ms", "max_ms"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.identity_claim is not False:
            raise ValueError("scale probe must not claim identity")
        if self.authorizes_action is not False:
            raise ValueError("scale probe must remain non-authorizing")
        if self.performance_claim is not False:
            raise ValueError("scale probe timing must not become a performance claim")


def _values(seed: int, dimensions: int) -> tuple[float, ...]:
    weights = tuple(((seed * 31 + index * 17) % 97) + 1 for index in range(dimensions))
    total = float(sum(weights))
    return tuple(value / total for value in weights)


def _descriptor(observation_id: str, seed: int, dimensions: int) -> AppearanceDescriptor:
    record = ForensicRecord(
        observation_id=observation_id,
        source_id=f"synthetic-source-{seed}",
        timestamp_ms=seed,
        category="person",
        confidence=1.0,
        evidence=ForensicEvidenceLink(
            event_id=f"synthetic-event-{seed}",
            producer="appearance-scale-probe",
            producer_version="1.0.0",
            config_sha256="1" * 64,
            source_revision="synthetic-v1",
        ),
    )
    return AppearanceDescriptor(
        schema="analytics.appearance-descriptor.v1",
        kind="model_free.color_histogram.v1",
        record=record,
        values=_values(seed, dimensions),
    )


def run_appearance_scale_probe(
    config: AppearanceScaleProbeConfig,
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> AppearanceScaleProbeResult:
    if not isinstance(config, AppearanceScaleProbeConfig):
        raise ValueError("config must be AppearanceScaleProbeConfig")
    if not callable(clock_ns):
        raise ValueError("clock_ns must be callable")

    probe = _descriptor("probe", 0, config.dimensions)
    candidates = tuple(
        _descriptor(f"candidate-{index}", index + 1, config.dimensions)
        for index in range(config.candidate_count)
    )

    timings: list[float] = []
    result_count: int | None = None
    for _ in range(config.iterations):
        start = clock_ns()
        matches = rank_appearance_similarity(
            probe,
            candidates,
            min_similarity=0.0,
            limit=min(config.result_limit, config.candidate_count),
        )
        end = clock_ns()
        if type(start) is not int or type(end) is not int or end < start:
            raise ValueError("clock_ns returned an invalid monotonic interval")
        timings.append((end - start) / 1_000_000.0)
        if result_count is None:
            result_count = len(matches)
        elif len(matches) != result_count:
            raise RuntimeError("appearance scale probe produced nondeterministic result count")

    elapsed = tuple(timings)
    return AppearanceScaleProbeResult(
        candidate_count=config.candidate_count,
        dimensions=config.dimensions,
        iterations=config.iterations,
        result_limit=config.result_limit,
        result_count=0 if result_count is None else result_count,
        elapsed_ms=elapsed,
        median_ms=float(median(elapsed)),
        max_ms=max(elapsed),
    )
