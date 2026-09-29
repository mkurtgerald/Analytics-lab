import unittest

from analytics_lab.forensic_appearance import AppearanceMatch
from analytics_lab.forensic_handoff import (
    envelope_for_appearance_matches,
    parse_envelope_json,
)
from analytics_lab.forensic_search import ForensicEvidenceLink, ForensicRecord


def _evidence(suffix: str) -> ForensicEvidenceLink:
    return ForensicEvidenceLink(
        event_id=f"event-{suffix}",
        producer="appearance-handoff-test",
        producer_version="1.0.0",
        config_sha256="1" * 64,
        source_revision="source-rev-1",
    )


def _record(
    observation_id: str,
    source_id: str,
    timestamp_ms: int,
) -> ForensicRecord:
    return ForensicRecord(
        observation_id=observation_id,
        source_id=source_id,
        timestamp_ms=timestamp_ms,
        category="person",
        confidence=0.9,
        evidence=_evidence(observation_id),
    )


def _match(
    candidate_id: str,
    similarity: float,
    *,
    source_id: str,
    timestamp_ms: int,
    probe_id: str = "probe",
) -> AppearanceMatch:
    return AppearanceMatch(
        probe_observation_id=probe_id,
        candidate=_record(candidate_id, source_id, timestamp_ms),
        similarity=similarity,
    )


class ForensicAppearanceHandoffTests(unittest.TestCase):
    def test_serializes_ordered_matches_into_existing_forensic_schema(self):
        matches = (
            _match("a", 0.95, source_id="camera-2", timestamp_ms=1100),
            _match("b", 0.90, source_id="camera-3", timestamp_ms=1200),
        )
        envelope = envelope_for_appearance_matches(
            matches,
            pre_roll_ms=3000,
            post_roll_ms=7000,
        )
        parsed = parse_envelope_json(envelope.to_json())
        self.assertEqual(parsed.schema, "analytics.forensic-result.v1")
        self.assertEqual(parsed.payload["kind"], "appearance_matches")
        self.assertEqual(parsed.payload["probe_observation_id"], "probe")
        self.assertEqual(
            parsed.payload["descriptor_schema"],
            "analytics.appearance-descriptor.v1",
        )
        self.assertEqual(
            parsed.payload["descriptor_kind"],
            "model_free.color_histogram.v1",
        )
        self.assertEqual(
            [item["similarity"] for item in parsed.payload["matches"]],
            [0.95, 0.90],
        )
        first = parsed.payload["matches"][0]
        self.assertEqual(first["hit"]["source_id"], "camera-2")
        self.assertEqual(first["hit"]["evidence"]["event_id"], "event-a")
        self.assertEqual(first["hit"]["playback"]["pre_roll_ms"], 3000)
        self.assertEqual(first["hit"]["playback"]["post_roll_ms"], 7000)
        self.assertFalse(first["identity_claim"])
        self.assertFalse(first["authorizes_action"])
        self.assertFalse(parsed.payload["identity_claim"])
        self.assertFalse(parsed.payload["authorizes_action"])

    def test_rejects_empty_match_tuple(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            envelope_for_appearance_matches(())

    def test_rejects_mixed_probe_ids(self):
        matches = (
            _match("a", 0.95, source_id="camera-2", timestamp_ms=1100, probe_id="p1"),
            _match("b", 0.90, source_id="camera-3", timestamp_ms=1200, probe_id="p2"),
        )
        with self.assertRaisesRegex(ValueError, "share one probe"):
            envelope_for_appearance_matches(matches)

    def test_rejects_duplicate_candidate_ids(self):
        matches = (
            _match("same", 0.95, source_id="camera-2", timestamp_ms=1100),
            _match("same", 0.90, source_id="camera-3", timestamp_ms=1200),
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            envelope_for_appearance_matches(matches)

    def test_rejects_nondeterministic_match_order(self):
        matches = (
            _match("low", 0.80, source_id="camera-2", timestamp_ms=1100),
            _match("high", 0.90, source_id="camera-3", timestamp_ms=1200),
        )
        with self.assertRaisesRegex(ValueError, "deterministic ranking order"):
            envelope_for_appearance_matches(matches)

    def test_equal_similarity_tie_order_is_deterministic(self):
        wrong = (
            _match("b", 0.90, source_id="camera-2", timestamp_ms=1000),
            _match("a", 0.90, source_id="camera-1", timestamp_ms=1000),
        )
        with self.assertRaisesRegex(ValueError, "deterministic ranking order"):
            envelope_for_appearance_matches(wrong)

        right = (
            _match("a", 0.90, source_id="camera-1", timestamp_ms=1000),
            _match("b", 0.90, source_id="camera-2", timestamp_ms=1000),
        )
        parsed = parse_envelope_json(envelope_for_appearance_matches(right).to_json())
        self.assertEqual(
            [item["hit"]["observation_id"] for item in parsed.payload["matches"]],
            ["a", "b"],
        )

    def test_match_count_is_bounded(self):
        matches = tuple(
            _match(
                f"candidate-{index}",
                1.0 - (index / 2000.0),
                source_id=f"camera-{index}",
                timestamp_ms=1000 + index,
            )
            for index in range(1001)
        )
        with self.assertRaisesRegex(ValueError, "result bound"):
            envelope_for_appearance_matches(matches)


if __name__ == "__main__":
    unittest.main()
