import unittest

from analytics_lab.forensic_handoff import (
    envelope_for_hit,
    envelope_for_trail,
    parse_envelope_json,
)
from analytics_lab.forensic_results import (
    operator_hit_from_record,
    operator_result_from_attribute_trail,
)
from analytics_lab.forensic_search import (
    AttributeFilter,
    ForensicAttribute,
    ForensicEvidenceLink,
    ForensicRecord,
)
from analytics_lab.forensic_temporal import AttributeCandidateTrail


def _evidence(suffix: str) -> ForensicEvidenceLink:
    return ForensicEvidenceLink(
        event_id=f"event-{suffix}",
        producer="analytics-test",
        producer_version="1.0.0",
        config_sha256="1" * 64,
        model_sha256="2" * 64,
        source_revision="source-rev-1",
    )


def _person(observation_id: str, source_id: str, timestamp_ms: int) -> ForensicRecord:
    return ForensicRecord(
        observation_id=observation_id,
        source_id=source_id,
        timestamp_ms=timestamp_ms,
        category="person",
        confidence=0.88,
        attributes=(
            ForensicAttribute("upper_color", "blue", 0.9, "appearance-v1"),
        ),
        evidence=_evidence(observation_id),
    )


class ForensicHandoffTests(unittest.TestCase):
    def test_hit_envelope_is_deterministic_and_non_authorizing(self):
        hit = operator_hit_from_record(_person("a", "camera-1", 1000))
        env = envelope_for_hit(hit)
        raw1 = env.to_json()
        raw2 = env.to_json()
        self.assertEqual(raw1, raw2)
        parsed = parse_envelope_json(raw1)
        self.assertEqual(parsed.schema, "analytics.forensic-result.v1")
        self.assertEqual(parsed.payload["kind"], "hit")
        self.assertFalse(parsed.payload["identity_claim"])
        self.assertFalse(parsed.payload["authorizes_action"])
        self.assertFalse(parsed.payload["playback"]["authorizes_action"])
        self.assertEqual(parsed.payload["source_id"], "camera-1")
        self.assertEqual(parsed.payload["evidence"]["event_id"], "event-a")

    def test_trail_envelope_preserves_hits_and_exact_attributes(self):
        first = _person("a", "camera-1", 1000)
        second = _person("b", "camera-2", 1200)
        trail = AttributeCandidateTrail(
            category="person",
            required_attributes=(AttributeFilter("upper_color", "blue"),),
            records=(first, second),
        )
        result = operator_result_from_attribute_trail(trail)
        env = envelope_for_trail(result)
        parsed = parse_envelope_json(env.to_json())
        self.assertEqual(parsed.payload["kind"], "trail")
        self.assertEqual(parsed.payload["association_kind"], "exact_attribute_candidate")
        self.assertEqual(len(parsed.payload["hits"]), 2)
        self.assertEqual(
            parsed.payload["required_attributes"],
            [{"name": "upper_color", "value": "blue"}],
        )
        self.assertFalse(parsed.payload["identity_claim"])
        self.assertFalse(parsed.payload["authorizes_action"])

    def test_rejects_wrong_schema(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_envelope_json('{"schema":"analytics.forensic-result.v2","payload":{"identity_claim":false,"authorizes_action":false}}')

    def test_rejects_authorizing_payload(self):
        raw = '{"schema":"analytics.forensic-result.v1","payload":{"identity_claim":false,"authorizes_action":true}}'
        with self.assertRaisesRegex(ValueError, "non-authorizing"):
            parse_envelope_json(raw)

    def test_rejects_identity_claim(self):
        raw = '{"schema":"analytics.forensic-result.v1","payload":{"identity_claim":true,"authorizes_action":false}}'
        with self.assertRaisesRegex(ValueError, "claim identity"):
            parse_envelope_json(raw)

    def test_rejects_unexpected_top_level_fields(self):
        raw = '{"schema":"analytics.forensic-result.v1","payload":{"identity_claim":false,"authorizes_action":false},"extra":1}'
        with self.assertRaisesRegex(ValueError, "unexpected top-level"):
            parse_envelope_json(raw)

    def test_rejects_invalid_json(self):
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            parse_envelope_json("{")

    def test_rejects_oversized_payload(self):
        raw = "x" * 1_000_001
        with self.assertRaisesRegex(ValueError, "too large"):
            parse_envelope_json(raw)


if __name__ == "__main__":
    unittest.main()
