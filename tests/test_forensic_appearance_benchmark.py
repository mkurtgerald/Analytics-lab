import unittest

from analytics_lab.forensic_appearance import AppearanceDescriptor
from analytics_lab.forensic_appearance_benchmark import (
    AppearanceBenchmarkCase,
    AppearanceBenchmarkResult,
    run_appearance_benchmark_case,
    summarize_appearance_benchmark,
)
from analytics_lab.forensic_search import ForensicEvidenceLink, ForensicRecord


def _evidence(suffix: str) -> ForensicEvidenceLink:
    return ForensicEvidenceLink(
        event_id=f"event-{suffix}",
        producer="appearance-benchmark-test",
        producer_version="1.0.0",
        config_sha256="1" * 64,
        source_revision="source-rev-1",
    )


def _descriptor(
    observation_id: str,
    values: tuple[float, ...],
    *,
    category: str = "person",
    source_id: str = "camera-1",
    timestamp_ms: int = 1000,
) -> AppearanceDescriptor:
    record = ForensicRecord(
        observation_id=observation_id,
        source_id=source_id,
        timestamp_ms=timestamp_ms,
        category=category,
        confidence=0.9,
        evidence=_evidence(observation_id),
    )
    return AppearanceDescriptor(
        schema="analytics.appearance-descriptor.v1",
        kind="model_free.color_histogram.v1",
        record=record,
        values=values,
    )


class AppearanceBenchmarkTests(unittest.TestCase):
    def test_perfect_top_two_case(self):
        probe = _descriptor("probe", (0.6, 0.3, 0.1))
        rel1 = _descriptor("rel1", (0.55, 0.35, 0.10), source_id="camera-2", timestamp_ms=1100)
        rel2 = _descriptor("rel2", (0.50, 0.40, 0.10), source_id="camera-3", timestamp_ms=1200)
        far = _descriptor("far", (0.10, 0.30, 0.60), source_id="camera-4", timestamp_ms=1300)
        case = AppearanceBenchmarkCase(
            case_id="perfect",
            probe=probe,
            candidates=(far, rel2, rel1),
            relevant_observation_ids=("rel1", "rel2"),
        )
        result = run_appearance_benchmark_case(case, k=2)
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(result.relevant_count, 2)
        self.assertEqual(result.retrieved_count, 2)
        self.assertEqual(result.relevant_retrieved, 2)
        self.assertEqual(result.precision_at_k, 1.0)
        self.assertEqual(result.recall_at_k, 1.0)
        self.assertEqual(result.reciprocal_rank, 1.0)
        self.assertTrue(result.top1_relevant)
        self.assertFalse(result.identity_claim)
        self.assertFalse(result.authorizes_action)

    def test_reciprocal_rank_when_first_result_is_irrelevant(self):
        probe = _descriptor("probe", (0.50, 0.30, 0.20))
        irrelevant = _descriptor("irrelevant", (0.49, 0.31, 0.20), source_id="camera-2")
        relevant = _descriptor("relevant", (0.45, 0.35, 0.20), source_id="camera-3")
        case = AppearanceBenchmarkCase(
            case_id="rr-two",
            probe=probe,
            candidates=(relevant, irrelevant),
            relevant_observation_ids=("relevant",),
        )
        result = run_appearance_benchmark_case(case, k=2)
        self.assertEqual(result.reciprocal_rank, 0.5)
        self.assertFalse(result.top1_relevant)
        self.assertEqual(result.recall_at_k, 1.0)
        self.assertEqual(result.precision_at_k, 0.5)

    def test_threshold_can_yield_zero_results_without_claims(self):
        probe = _descriptor("probe", (0.7, 0.2, 0.1))
        relevant = _descriptor("relevant", (0.4, 0.3, 0.3), source_id="camera-2")
        case = AppearanceBenchmarkCase(
            case_id="threshold-empty",
            probe=probe,
            candidates=(relevant,),
            relevant_observation_ids=("relevant",),
        )
        result = run_appearance_benchmark_case(case, k=1, min_similarity=0.95)
        self.assertEqual(result.retrieved_count, 0)
        self.assertEqual(result.relevant_retrieved, 0)
        self.assertEqual(result.precision_at_k, 0.0)
        self.assertEqual(result.recall_at_k, 0.0)
        self.assertEqual(result.reciprocal_rank, 0.0)
        self.assertFalse(result.top1_relevant)

    def test_summary_averages_cases_deterministically(self):
        first = AppearanceBenchmarkResult(
            case_id="a",
            candidate_count=3,
            relevant_count=1,
            k=1,
            retrieved_count=1,
            relevant_retrieved=1,
            precision_at_k=1.0,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            top1_relevant=True,
        )
        second = AppearanceBenchmarkResult(
            case_id="b",
            candidate_count=3,
            relevant_count=1,
            k=1,
            retrieved_count=1,
            relevant_retrieved=0,
            precision_at_k=0.0,
            recall_at_k=0.0,
            reciprocal_rank=0.5,
            top1_relevant=False,
        )
        summary = summarize_appearance_benchmark((first, second))
        self.assertEqual(summary.case_count, 2)
        self.assertEqual(summary.mean_precision_at_k, 0.5)
        self.assertEqual(summary.mean_recall_at_k, 0.5)
        self.assertEqual(summary.mean_reciprocal_rank, 0.75)
        self.assertEqual(summary.top1_rate, 0.5)
        self.assertFalse(summary.identity_claim)
        self.assertFalse(summary.authorizes_action)

    def test_case_rejects_probe_inside_candidate_set(self):
        probe = _descriptor("probe", (0.5, 0.5))
        candidate = _descriptor("candidate", (0.5, 0.5), source_id="camera-2")
        with self.assertRaisesRegex(ValueError, "exclude the probe"):
            AppearanceBenchmarkCase(
                case_id="bad-probe",
                probe=probe,
                candidates=(probe, candidate),
                relevant_observation_ids=("candidate",),
            )

    def test_case_rejects_unknown_relevance_label(self):
        probe = _descriptor("probe", (0.5, 0.5))
        candidate = _descriptor("candidate", (0.5, 0.5), source_id="camera-2")
        with self.assertRaisesRegex(ValueError, "not a candidate"):
            AppearanceBenchmarkCase(
                case_id="bad-relevance",
                probe=probe,
                candidates=(candidate,),
                relevant_observation_ids=("missing",),
            )

    def test_case_rejects_duplicate_relevance_ids(self):
        probe = _descriptor("probe", (0.5, 0.5))
        candidate = _descriptor("candidate", (0.5, 0.5), source_id="camera-2")
        with self.assertRaisesRegex(ValueError, "must be unique"):
            AppearanceBenchmarkCase(
                case_id="dup-rel",
                probe=probe,
                candidates=(candidate,),
                relevant_observation_ids=("candidate", "candidate"),
            )

    def test_case_rejects_category_mismatch(self):
        probe = _descriptor("probe", (0.5, 0.5), category="person")
        vehicle = _descriptor("vehicle", (0.5, 0.5), category="vehicle")
        with self.assertRaisesRegex(ValueError, "match probe category"):
            AppearanceBenchmarkCase(
                case_id="category",
                probe=probe,
                candidates=(vehicle,),
                relevant_observation_ids=("vehicle",),
            )

    def test_case_rejects_dimension_mismatch(self):
        probe = _descriptor("probe", (0.5, 0.5))
        candidate = _descriptor("candidate", (0.2, 0.3, 0.5), source_id="camera-2")
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            AppearanceBenchmarkCase(
                case_id="dims",
                probe=probe,
                candidates=(candidate,),
                relevant_observation_ids=("candidate",),
            )

    def test_summary_rejects_duplicate_case_ids(self):
        result = AppearanceBenchmarkResult(
            case_id="same",
            candidate_count=1,
            relevant_count=1,
            k=1,
            retrieved_count=1,
            relevant_retrieved=1,
            precision_at_k=1.0,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            top1_relevant=True,
        )
        with self.assertRaisesRegex(ValueError, "case IDs must be unique"):
            summarize_appearance_benchmark((result, result))

    def test_summary_case_count_is_bounded(self):
        results = tuple(
            AppearanceBenchmarkResult(
                case_id=f"case-{index}",
                candidate_count=1,
                relevant_count=1,
                k=1,
                retrieved_count=1,
                relevant_retrieved=1,
                precision_at_k=1.0,
                recall_at_k=1.0,
                reciprocal_rank=1.0,
                top1_relevant=True,
            )
            for index in range(101)
        )
        with self.assertRaisesRegex(RuntimeError, "case count exceeds"):
            summarize_appearance_benchmark(results)


if __name__ == "__main__":
    unittest.main()
