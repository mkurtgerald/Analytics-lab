"""Valid descriptor tolerance must remain compatible with the operator wire score."""
from dataclasses import asdict
import math
import unittest
from unittest.mock import patch

from analytics_lab.forensic_appearance import (
    AppearanceDescriptor, AppearanceMatch, rank_appearance_similarity,
)
from analytics_lab.forensic_handoff import envelope_for_appearance_matches, parse_envelope_json
from analytics_lab.forensic_search import ForensicEvidenceLink, ForensicRecord


def descriptor(name, values, *, source="synthetic_camera", timestamp=1000):
    record = ForensicRecord(
        name, source, timestamp, "person", 0.9,
        evidence=ForensicEvidenceLink(name + "_event", "synthetic_producer", "1", "a" * 64),
    )
    return AppearanceDescriptor("analytics.appearance-descriptor.v1",
                                "model_free.color_histogram.v1", record, values)


class AppearanceScoreBoundaryTests(unittest.TestCase):
    def test_valid_l1_excess_reaches_operator_handoff_without_losing_evidence(self):
        for values in ((0.5, 0.5000000001), (0.0, math.nextafter(1.0 + 1e-9, 1.0)),
                       (0.5, 0.5000000001) + (0.0,) * 510):
            with self.subTest(dimensions=len(values), total=sum(values)):
                probe = descriptor("probe", values)
                candidate = descriptor("candidate", values, source="synthetic_camera_2")
                matches = rank_appearance_similarity(probe, (candidate,), min_similarity=1.0)
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0].similarity, 1.0)
                self.assertEqual(candidate.values, values)
                self.assertIs(matches[0].candidate, candidate.record)
                raw = envelope_for_appearance_matches(matches).to_json()
                envelope = parse_envelope_json(raw)
                self.assertEqual(envelope.to_json(), raw)
                match = envelope.payload["matches"][0]
                self.assertEqual(match["similarity"], 1.0)
                self.assertEqual(match["hit"]["evidence"], asdict(candidate.record.evidence))
                self.assertEqual(match["hit"]["timestamp_ms"], candidate.record.timestamp_ms)
                self.assertEqual(match["hit"]["playback"]["source_id"], candidate.record.source_id)
                for item in (envelope.payload, match, match["hit"]):
                    self.assertIs(item["identity_claim"], False)
                    self.assertIs(item["authorizes_action"], False)
                self.assertIs(match["hit"]["playback"]["authorizes_action"], False)

    def test_saturated_scores_use_existing_deterministic_tie_order(self):
        probe = descriptor("probe", (0.5, 0.5000000006))
        later = descriptor("later", (0.5, 0.5000000006), timestamp=1001)
        earlier = descriptor("earlier", (0.5, 0.5000000002), timestamp=1000)
        matches = rank_appearance_similarity(probe, (later, earlier))
        self.assertEqual([item.candidate.observation_id for item in matches], ["earlier", "later"])
        self.assertEqual([item.similarity for item in matches], [1.0, 1.0])
        self.assertEqual(len(parse_envelope_json(envelope_for_appearance_matches(matches).to_json())
                             .payload["matches"]), 2)

    def test_ordinary_scores_and_inclusive_threshold_are_unchanged(self):
        probe = descriptor("probe", (0.6, 0.4))
        candidate = descriptor("candidate", (0.4, 0.6))
        matches = rank_appearance_similarity(probe, (candidate,), min_similarity=0.8)
        self.assertEqual(matches[0].similarity, 0.8)
        self.assertEqual(rank_appearance_similarity(probe, (candidate,), min_similarity=0.81), ())

    def test_invalid_descriptor_values_and_explicit_out_of_range_scores_stay_rejected(self):
        for values in ((0.5, 0.50000001), (0.5, 0.49999999), (-0.1, 1.1),
                       (float("nan"), 1.0), (float("inf"), 0.0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                descriptor("invalid", values)
        candidate = descriptor("candidate", (0.5, 0.5)).record
        for score in (1.0000000001, 1.1, float("nan"), float("inf"), -0.1):
            with self.subTest(score=score), self.assertRaises(ValueError):
                AppearanceMatch("probe", candidate, score)

    def test_scorer_does_not_hide_material_or_nonfinite_result_errors(self):
        probe = descriptor("probe", (0.5, 0.5))
        candidate = descriptor("candidate", (0.5, 0.5))
        for score in (1.00000001, 2.0, float("nan"), float("inf")):
            with self.subTest(score=score), patch(
                "analytics_lab.forensic_appearance.histogram_intersection", return_value=score
            ), self.assertRaises(ValueError):
                rank_appearance_similarity(probe, (candidate,))


if __name__ == "__main__":
    unittest.main()
