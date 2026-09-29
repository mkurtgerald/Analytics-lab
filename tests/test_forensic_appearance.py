import math
import unittest

from analytics_lab.forensic_appearance import (
    AppearanceDescriptor,
    rank_appearance_similarity,
)
from analytics_lab.forensic_search import ForensicEvidenceLink, ForensicRecord


def _evidence(suffix: str) -> ForensicEvidenceLink:
    return ForensicEvidenceLink(
        event_id=f"event-{suffix}",
        producer="appearance-test",
        producer_version="1.0.0",
        config_sha256="1" * 64,
        source_revision="source-rev-1",
    )


def _record(
    observation_id: str,
    *,
    source_id: str = "camera-1",
    timestamp_ms: int = 1000,
    category: str = "person",
) -> ForensicRecord:
    return ForensicRecord(
        observation_id=observation_id,
        source_id=source_id,
        timestamp_ms=timestamp_ms,
        category=category,
        confidence=0.9,
        evidence=_evidence(observation_id),
    )


def _descriptor(
    observation_id: str,
    values: tuple[float, ...],
    *,
    source_id: str = "camera-1",
    timestamp_ms: int = 1000,
    category: str = "person",
) -> AppearanceDescriptor:
    return AppearanceDescriptor(
        schema="analytics.appearance-descriptor.v1",
        kind="model_free.color_histogram.v1",
        record=_record(
            observation_id,
            source_id=source_id,
            timestamp_ms=timestamp_ms,
            category=category,
        ),
        values=values,
    )


class ForensicAppearanceTests(unittest.TestCase):
    def test_near_candidate_ranks_above_far_candidate(self):
        probe = _descriptor("probe", (0.6, 0.3, 0.1))
        near = _descriptor("near", (0.5, 0.4, 0.1), source_id="camera-2", timestamp_ms=1100)
        far = _descriptor("far", (0.1, 0.3, 0.6), source_id="camera-3", timestamp_ms=1200)
        result = rank_appearance_similarity(probe, (far, near))
        self.assertEqual([item.candidate.observation_id for item in result], ["near", "far"])
        self.assertTrue(result[0].similarity > result[1].similarity)

    def test_inclusive_threshold_tolerates_binary64_boundary(self):
        probe = _descriptor("probe", (0.4, 0.3, 0.2, 0.1))
        boundary = _descriptor("boundary", (0.3, 0.3, 0.2, 0.2), source_id="camera-2")
        result = rank_appearance_similarity(
            probe,
            (boundary,),
            min_similarity=0.9,
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(math.isclose(result[0].similarity, 0.9, abs_tol=1e-9))

    def test_deterministic_tie_ordering(self):
        probe = _descriptor("probe", (0.5, 0.5))
        b = _descriptor("b", (0.5, 0.5), source_id="camera-2", timestamp_ms=1000)
        a = _descriptor("a", (0.5, 0.5), source_id="camera-1", timestamp_ms=1000)
        result = rank_appearance_similarity(probe, (b, a))
        self.assertEqual([item.candidate.observation_id for item in result], ["a", "b"])

    def test_probe_is_excluded(self):
        probe = _descriptor("same", (0.5, 0.5))
        result = rank_appearance_similarity(probe, (probe,))
        self.assertEqual(result, ())

    def test_result_limit_is_enforced(self):
        probe = _descriptor("probe", (0.5, 0.5))
        candidates = tuple(
            _descriptor(f"c{i}", (0.5, 0.5), source_id=f"camera-{i+2}", timestamp_ms=1000+i)
            for i in range(5)
        )
        result = rank_appearance_similarity(probe, candidates, limit=2)
        self.assertEqual(len(result), 2)

    def test_face_and_license_plate_categories_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "category"):
            _descriptor("face", (0.5, 0.5), category="face")
        with self.assertRaisesRegex(ValueError, "category"):
            _descriptor("plate", (0.5, 0.5), category="license_plate")

    def test_learned_descriptor_kind_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "kind"):
            AppearanceDescriptor(
                schema="analytics.appearance-descriptor.v1",
                kind="learned.reid.v1",
                record=_record("x"),
                values=(0.5, 0.5),
            )

    def test_invalid_histograms_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "L1 normalized"):
            _descriptor("bad-sum", (0.2, 0.2))
        with self.assertRaisesRegex(ValueError, "finite"):
            _descriptor("nan", (float("nan"), 1.0))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _descriptor("negative", (-0.1, 1.1))

    def test_category_mismatch_is_rejected(self):
        probe = _descriptor("probe", (0.5, 0.5), category="person")
        vehicle = _descriptor("vehicle", (0.5, 0.5), category="vehicle")
        with self.assertRaisesRegex(ValueError, "category mismatch"):
            rank_appearance_similarity(probe, (vehicle,))

    def test_dimension_mismatch_is_rejected(self):
        probe = _descriptor("probe", (0.5, 0.5))
        candidate = _descriptor("candidate", (0.3, 0.3, 0.4))
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            rank_appearance_similarity(probe, (candidate,))

    def test_duplicate_candidate_ids_are_rejected(self):
        probe = _descriptor("probe", (0.5, 0.5))
        a = _descriptor("dup", (0.5, 0.5), source_id="camera-2")
        b = _descriptor("dup", (0.5, 0.5), source_id="camera-3")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rank_appearance_similarity(probe, (a, b))

    def test_candidate_count_is_bounded(self):
        probe = _descriptor("probe", (0.5, 0.5))
        candidate = _descriptor("c", (0.5, 0.5), source_id="camera-2")
        with self.assertRaisesRegex(RuntimeError, "count exceeds"):
            rank_appearance_similarity(probe, (candidate,) * 10_001)

    def test_matches_remain_nonidentity_and_nonauthorizing(self):
        probe = _descriptor("probe", (0.5, 0.5))
        candidate = _descriptor("candidate", (0.5, 0.5), source_id="camera-2")
        match = rank_appearance_similarity(probe, (candidate,))[0]
        self.assertFalse(match.identity_claim)
        self.assertFalse(match.authorizes_action)
        self.assertEqual(match.candidate.evidence, candidate.record.evidence)


if __name__ == "__main__":
    unittest.main()
