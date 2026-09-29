"""Deterministic engineering benchmark harness for model-free appearance ranking.

Benchmark relevance is project-authored fixture relevance only (for example,
same synthetic color-distribution family). It is never a person/object identity
label and does not support biometric/ReID claims or commercial accuracy claims.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .forensic_appearance import (
    AppearanceDescriptor,
    AppearanceMatch,
    rank_appearance_similarity,
)

_MAX_CASES = 100
_MAX_K = 1000


def _unit_interval(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


@dataclass(frozen=True)
class AppearanceBenchmarkCase:
    """One deterministic fixture-defined appearance-ranking benchmark case."""

    case_id: str
    probe: AppearanceDescriptor
    candidates: tuple[AppearanceDescriptor, ...]
    relevant_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id or len(self.case_id) > 128:
            raise ValueError("case_id must contain 1-128 characters")
        if not isinstance(self.probe, AppearanceDescriptor):
            raise ValueError("probe must be AppearanceDescriptor")
        if type(self.candidates) is not tuple or not self.candidates:
            raise ValueError("candidates must be a nonempty immutable tuple")
        if any(not isinstance(item, AppearanceDescriptor) for item in self.candidates):
            raise ValueError("candidates must contain AppearanceDescriptor values")

        candidate_ids = tuple(item.record.observation_id for item in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("benchmark candidate observation IDs must be unique")
        if self.probe.record.observation_id in candidate_ids:
            raise ValueError("benchmark candidates must exclude the probe observation")

        if type(self.relevant_observation_ids) is not tuple or not self.relevant_observation_ids:
            raise ValueError("benchmark case requires at least one fixture-relevant candidate")
        if len(self.relevant_observation_ids) != len(set(self.relevant_observation_ids)):
            raise ValueError("fixture-relevant observation IDs must be unique")
        unknown = set(self.relevant_observation_ids) - set(candidate_ids)
        if unknown:
            raise ValueError("fixture-relevant observation ID is not a candidate")

        if any(item.record.category != self.probe.record.category for item in self.candidates):
            raise ValueError("benchmark candidates must match probe category")
        if any(item.kind != self.probe.kind or item.schema != self.probe.schema for item in self.candidates):
            raise ValueError("benchmark descriptor kind/schema mismatch")
        if any(len(item.values) != len(self.probe.values) for item in self.candidates):
            raise ValueError("benchmark descriptor dimension mismatch")


@dataclass(frozen=True)
class AppearanceBenchmarkResult:
    case_id: str
    candidate_count: int
    relevant_count: int
    k: int
    retrieved_count: int
    relevant_retrieved: int
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    top1_relevant: bool
    identity_claim: bool = False
    authorizes_action: bool = False

    def __post_init__(self) -> None:
        for name in ("candidate_count", "relevant_count", "k", "retrieved_count", "relevant_retrieved"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.k < 1:
            raise ValueError("k must be at least 1")
        for name in ("precision_at_k", "recall_at_k", "reciprocal_rank"):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))
        if type(self.top1_relevant) is not bool:
            raise ValueError("top1_relevant must be bool")
        if self.identity_claim is not False:
            raise ValueError("benchmark result must not claim identity")
        if self.authorizes_action is not False:
            raise ValueError("benchmark result must remain non-authorizing")


@dataclass(frozen=True)
class AppearanceBenchmarkSummary:
    case_count: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    top1_rate: float
    identity_claim: bool = False
    authorizes_action: bool = False

    def __post_init__(self) -> None:
        if type(self.case_count) is not int or self.case_count < 1:
            raise ValueError("case_count must be an integer >= 1")
        for name in (
            "mean_precision_at_k",
            "mean_recall_at_k",
            "mean_reciprocal_rank",
            "top1_rate",
        ):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))
        if self.identity_claim is not False:
            raise ValueError("benchmark summary must not claim identity")
        if self.authorizes_action is not False:
            raise ValueError("benchmark summary must remain non-authorizing")


def run_appearance_benchmark_case(
    case: AppearanceBenchmarkCase,
    *,
    k: int = 5,
    min_similarity: float = 0.0,
) -> AppearanceBenchmarkResult:
    if not isinstance(case, AppearanceBenchmarkCase):
        raise ValueError("case must be AppearanceBenchmarkCase")
    if type(k) is not int or not 1 <= k <= _MAX_K:
        raise ValueError("k must be an integer in [1, 1000]")

    ranked: tuple[AppearanceMatch, ...] = rank_appearance_similarity(
        case.probe,
        case.candidates,
        min_similarity=min_similarity,
        limit=min(len(case.candidates), 1000),
    )
    top = ranked[:k]
    relevant = set(case.relevant_observation_ids)
    top_ids = [item.candidate.observation_id for item in top]
    relevant_retrieved = sum(1 for observation_id in top_ids if observation_id in relevant)
    precision = relevant_retrieved / len(top) if top else 0.0
    recall = relevant_retrieved / len(relevant)

    first_relevant_rank = next(
        (
            index
            for index, match in enumerate(ranked, start=1)
            if match.candidate.observation_id in relevant
        ),
        None,
    )
    reciprocal_rank = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank

    return AppearanceBenchmarkResult(
        case_id=case.case_id,
        candidate_count=len(case.candidates),
        relevant_count=len(relevant),
        k=k,
        retrieved_count=len(top),
        relevant_retrieved=relevant_retrieved,
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        top1_relevant=bool(top_ids and top_ids[0] in relevant),
    )


def summarize_appearance_benchmark(
    results: Iterable[AppearanceBenchmarkResult],
) -> AppearanceBenchmarkSummary:
    try:
        items = tuple(results)
    except TypeError as exc:
        raise ValueError("results must be iterable") from exc
    if not items:
        raise ValueError("benchmark summary requires at least one result")
    if len(items) > _MAX_CASES:
        raise RuntimeError("appearance benchmark case count exceeds supported bound")
    if any(not isinstance(item, AppearanceBenchmarkResult) for item in items):
        raise ValueError("results must contain AppearanceBenchmarkResult values")
    case_ids = [item.case_id for item in items]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark result case IDs must be unique")

    count = len(items)
    return AppearanceBenchmarkSummary(
        case_count=count,
        mean_precision_at_k=sum(item.precision_at_k for item in items) / count,
        mean_recall_at_k=sum(item.recall_at_k for item in items) / count,
        mean_reciprocal_rank=sum(item.reciprocal_rank for item in items) / count,
        top1_rate=sum(1.0 if item.top1_relevant else 0.0 for item in items) / count,
    )
