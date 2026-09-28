import unittest

from analytics_lab.forensic_results import (
    ForensicOperatorHit,
    ForensicTrailResult,
    PlaybackReference,
    operator_hit_from_record,
    operator_result_from_attribute_trail,
    operator_result_from_plate_trail,
)
from analytics_lab.forensic_search import (
    AttributeFilter,
    ForensicAttribute,
    ForensicEvidenceLink,
    ForensicRecord,
)
from analytics_lab.forensic_temporal import AttributeCandidateTrail, TemporalTrail


def _evidence(suffix: str) -> ForensicEvidenceLink:
    return ForensicEvidenceLink(
        event_id=f"event-{suffix}",
        producer="analytics-test",
        producer_version="1.0.0",
        config_sha256="1" * 64,
        model_sha256="2" * 64,
        source_revision="source-rev-1",
    )


def _plate(observation_id: str, source_id: str, timestamp_ms: int) -> ForensicRecord:
    return ForensicRecord(
        observation_id=observation_id,
        source_id=source_id,
        timestamp_ms=timestamp_ms,
        category="license_plate",
        confidence=0.9,
        plate_text="ABC123",
        evidence=_evidence(observation_id),
    )


def _person(observation_id: str, source_id: str, timestamp_ms: int) -> ForensicRecord:
    return ForensicRecord(
        observation_id=observation_id,
        source_id=source_id,
        timestamp_ms=timestamp_ms,
        category="person",
        confidence=0.88,
        track_id=f"session:{observation_id}",
        attributes=(
            ForensicAttribute("upper_color", "blue", 0.9, "appearance-v1"),
            ForensicAttribute("direction", "east", 0.8, "motion-v1"),
        ),
        evidence=_evidence(observation_id),
    )


class ForensicResultsTests(unittest.TestCase):
    def test_single_hit_binds_evidence_and_non_authorizing_playback(self):
        record = _person("a", "camera-1", 1000)
        hit = operator_hit_from_record(
            record,
            pre_roll_ms=3000,
            post_roll_ms=7000,
        )
        self.assertEqual(hit.observation_id, record.observation_id)
        self.assertEqual(hit.evidence, record.evidence)
        self.assertEqual(hit.playback.source_id, record.source_id)
        self.assertEqual(hit.playback.timestamp_ms, record.timestamp_ms)
        self.assertEqual(hit.playback.pre_roll_ms, 3000)
        self.assertEqual(hit.playback.post_roll_ms, 7000)
        self.assertFalse(hit.identity_claim)
        self.assertFalse(hit.authorizes_action)
        self.assertFalse(hit.playback.authorizes_action)
        self.assertEqual(hit.local_track_id, record.track_id)
        self.assertEqual(hit.attributes, record.attributes)

    def test_playback_roll_is_bounded(self):
        evidence = _evidence("a")
        with self.assertRaisesRegex(ValueError, "pre_roll_ms"):
            PlaybackReference("camera-1", 1000, evidence, pre_roll_ms=-1)
        with self.assertRaisesRegex(ValueError, "post_roll_ms"):
            PlaybackReference("camera-1", 1000, evidence, post_roll_ms=300_001)

    def test_operator_hit_rejects_mismatched_playback_binding(self):
        evidence = _evidence("a")
        playback = PlaybackReference("camera-2", 1000, evidence)
        with self.assertRaisesRegex(ValueError, "same hit/evidence"):
            ForensicOperatorHit(
                observation_id="a",
                source_id="camera-1",
                timestamp_ms=1000,
                category="person",
                confidence=0.9,
                evidence=evidence,
                playback=playback,
            )

    def test_plate_trail_projection_preserves_exact_plate_and_evidence(self):
        first = _plate("a", "camera-1", 1000)
        second = _plate("b", "camera-2", 1200)
        trail = TemporalTrail(
            association_kind="exact_plate_text",
            association_value="ABC123",
            records=(first, second),
        )
        result = operator_result_from_plate_trail(trail)
        self.assertEqual(result.association_kind, "exact_plate_text")
        self.assertEqual(result.association_value, "ABC123")
        self.assertEqual([hit.plate_text for hit in result.hits], ["ABC123", "ABC123"])
        self.assertEqual(result.hits[0].evidence, first.evidence)
        self.assertEqual(result.hits[1].evidence, second.evidence)
        self.assertFalse(result.identity_claim)
        self.assertFalse(result.authorizes_action)

    def test_attribute_trail_projection_preserves_filters_and_attributes(self):
        first = _person("a", "camera-1", 1000)
        second = _person("b", "camera-2", 1200)
        required = (
            AttributeFilter("direction", "east"),
            AttributeFilter("upper_color", "blue"),
        )
        trail = AttributeCandidateTrail(
            category="person",
            required_attributes=required,
            records=(first, second),
        )
        result = operator_result_from_attribute_trail(trail)
        self.assertEqual(result.association_kind, "exact_attribute_candidate")
        self.assertIsNone(result.association_value)
        self.assertEqual(result.required_attributes, required)
        self.assertEqual(result.hits[0].attributes, first.attributes)
        self.assertEqual(result.hits[1].attributes, second.attributes)
        self.assertFalse(result.identity_claim)
        self.assertFalse(result.authorizes_action)

    def test_plate_result_cannot_carry_attribute_filters(self):
        first = operator_hit_from_record(_plate("a", "camera-1", 1000))
        second = operator_hit_from_record(_plate("b", "camera-2", 1200))
        with self.assertRaisesRegex(ValueError, "cannot carry attribute"):
            ForensicTrailResult(
                association_kind="exact_plate_text",
                association_value="ABC123",
                required_attributes=(AttributeFilter("color", "red"),),
                hits=(first, second),
            )

    def test_attribute_result_requires_canonical_deduplicated_filters(self):
        first = operator_hit_from_record(_person("a", "camera-1", 1000))
        second = operator_hit_from_record(_person("b", "camera-2", 1200))
        with self.assertRaisesRegex(ValueError, "canonical"):
            ForensicTrailResult(
                association_kind="exact_attribute_candidate",
                required_attributes=(
                    AttributeFilter("upper_color", "blue"),
                    AttributeFilter("direction", "east"),
                ),
                hits=(first, second),
            )

    def test_trail_result_requires_distinct_sources(self):
        first = operator_hit_from_record(_person("a", "camera-1", 1000))
        second = operator_hit_from_record(_person("b", "camera-1", 1200))
        with self.assertRaisesRegex(ValueError, "distinct sources"):
            ForensicTrailResult(
                association_kind="exact_attribute_candidate",
                required_attributes=(AttributeFilter("upper_color", "blue"),),
                hits=(first, second),
            )

    def test_projection_never_authorizes_action(self):
        first = operator_hit_from_record(_person("a", "camera-1", 1000))
        second = operator_hit_from_record(_person("b", "camera-2", 1200))
        with self.assertRaisesRegex(ValueError, "non-authorizing"):
            ForensicTrailResult(
                association_kind="exact_attribute_candidate",
                required_attributes=(AttributeFilter("upper_color", "blue"),),
                hits=(first, second),
                authorizes_action=True,
            )


if __name__ == "__main__":
    unittest.main()
