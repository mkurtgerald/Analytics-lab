import unittest
from analytics_lab.evaluation import EvaluationConfig, LabeledPersonDown, evaluate_person_down_candidates


def event(start, end):
    return {
        "event_type": "person_down_candidate",
        "status": "candidate",
        "evidence": [{"type": "observation_window", "start_timestamp_ms": start, "end_timestamp_ms": end}],
    }


class EvaluationTests(unittest.TestCase):
    def test_one_to_one_matching_prevents_duplicate_recall_inflation(self):
        result = evaluate_person_down_candidates(
            [event(1_000, 4_000), event(1_500, 4_500)],
            [LabeledPersonDown(1_000, 5_000, "fall-a")],
            video_start_timestamp_ms=0, video_end_timestamp_ms=3_600_000,
        )
        self.assertEqual(result.matched_events, 1)
        self.assertEqual(result.false_alerts, 1)
        self.assertEqual(result.missed_episodes, 0)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 0.5)
        self.assertEqual(result.false_alerts_per_camera_hour, 1.0)

    def test_miss_and_normal_negative_metrics(self):
        missed = evaluate_person_down_candidates(
            [], [LabeledPersonDown(10_000, 12_000, "a")],
            video_start_timestamp_ms=0, video_end_timestamp_ms=60_000,
        )
        self.assertEqual(missed.missed_episodes, 1)
        self.assertEqual(missed.recall, 0.0)
        self.assertIsNone(missed.precision)
        negative = evaluate_person_down_candidates(
            [event(5_000, 8_000)], [],
            video_start_timestamp_ms=0, video_end_timestamp_ms=1_800_000,
        )
        self.assertIsNone(negative.recall)
        self.assertEqual(negative.false_alerts, 1)
        self.assertEqual(negative.false_alerts_per_camera_hour, 2.0)

    def test_tolerance_and_delay(self):
        result = evaluate_person_down_candidates(
            [event(8_500, 9_500), event(19_000, 21_000)],
            [LabeledPersonDown(10_000, 12_000, "a"), LabeledPersonDown(20_000, 22_000, "b")],
            video_start_timestamp_ms=0, video_end_timestamp_ms=30_000,
            config=EvaluationConfig(match_tolerance_ms=1_000),
        )
        self.assertEqual(result.matched_label_ids, ("a", "b"))
        self.assertEqual(result.median_alert_delay_ms, 250.0)

    def test_deterministic_closest_label_match(self):
        result = evaluate_person_down_candidates(
            [event(4_000, 6_000)],
            [LabeledPersonDown(5_000, 7_000, "a"), LabeledPersonDown(6_000, 8_000, "b")],
            video_start_timestamp_ms=0, video_end_timestamp_ms=10_000,
            config=EvaluationConfig(match_tolerance_ms=2_000),
        )
        self.assertEqual(result.matched_label_ids, ("b",))
        self.assertEqual(result.missed_episodes, 1)

    def test_rejects_out_of_bounds_and_duplicate_labels(self):
        with self.assertRaises(ValueError):
            evaluate_person_down_candidates(
                [], [LabeledPersonDown(0, 5, "x"), LabeledPersonDown(6, 7, "x")],
                video_start_timestamp_ms=0, video_end_timestamp_ms=10,
            )
        with self.assertRaises(ValueError):
            evaluate_person_down_candidates([event(0, 11)], [], video_start_timestamp_ms=0, video_end_timestamp_ms=10)

    def test_rejects_malformed_events_and_nonpositive_duration(self):
        bad = {"event_type": "person_down_candidate", "status": "candidate", "evidence": []}
        with self.assertRaises(ValueError):
            evaluate_person_down_candidates([bad], [], video_start_timestamp_ms=0, video_end_timestamp_ms=10)
        with self.assertRaises(ValueError):
            evaluate_person_down_candidates([], [], video_start_timestamp_ms=1, video_end_timestamp_ms=1)

    def test_label_validation(self):
        with self.assertRaises(ValueError):
            LabeledPersonDown(10, 9, "x")
        with self.assertRaises(ValueError):
            LabeledPersonDown(0, 1, "")


if __name__ == "__main__":
    unittest.main()
