import unittest
from analytics_lab.evaluation import (
    EvaluationConfig, EvaluationSample, LabeledPersonDown,
    aggregate_person_down_evaluations, evaluate_person_down_candidates,
)


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
        self.assertEqual(result.alert_delays_ms, (3000,))

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

    def test_matching_maximizes_cardinality_before_delay(self):
        result = evaluate_person_down_candidates(
            [event(0, 1_000), event(1_000, 1_000)],
            [LabeledPersonDown(0, 0, "a"), LabeledPersonDown(1_000, 1_000, "b")],
            video_start_timestamp_ms=0, video_end_timestamp_ms=2_000,
            config=EvaluationConfig(match_tolerance_ms=0),
        )
        self.assertEqual(result.matched_events, 2)
        self.assertEqual(result.missed_episodes, 0)
        self.assertEqual(result.false_alerts, 0)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.matched_label_ids, ("a", "b"))
        self.assertEqual(result.alert_delays_ms, (1_000, 0))

    def test_deterministic_earliest_ending_label_match(self):
        result = evaluate_person_down_candidates(
            [event(4_000, 6_000)],
            [LabeledPersonDown(5_000, 7_000, "a"), LabeledPersonDown(6_000, 8_000, "b")],
            video_start_timestamp_ms=0, video_end_timestamp_ms=10_000,
            config=EvaluationConfig(match_tolerance_ms=2_000),
        )
        self.assertEqual(result.matched_label_ids, ("a",))
        self.assertEqual(result.missed_episodes, 1)
        self.assertEqual(result.alert_delays_ms, (1_000,))

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

    def test_aggregate_micro_metrics_and_breakdowns(self):
        first = evaluate_person_down_candidates(
            [event(1_000, 4_000), event(10_000, 11_000)],
            [LabeledPersonDown(1_000, 5_000, "a")],
            video_start_timestamp_ms=0, video_end_timestamp_ms=3_600_000,
        )
        second = evaluate_person_down_candidates(
            [], [LabeledPersonDown(3_700_000, 3_705_000, "b")],
            video_start_timestamp_ms=3_600_000, video_end_timestamp_ms=7_200_000,
        )
        third = evaluate_person_down_candidates(
            [event(1_000, 2_000)], [],
            video_start_timestamp_ms=0, video_end_timestamp_ms=1_800_000,
        )
        aggregate = aggregate_person_down_evaluations([
            EvaluationSample("s1", "site-a", "cam-1", 0, 3_600_000, first),
            EvaluationSample("s2", "site-a", "cam-1", 3_600_000, 7_200_000, second),
            EvaluationSample("s3", "site-b", "cam-2", 0, 1_800_000, third),
        ])
        self.assertEqual((aggregate.sample_count, aggregate.site_count, aggregate.camera_count), (3, 2, 2))
        self.assertEqual(aggregate.positive_episodes, 2)
        self.assertEqual(aggregate.candidate_events, 3)
        self.assertEqual(aggregate.matched_events, 1)
        self.assertEqual(aggregate.missed_episodes, 1)
        self.assertEqual(aggregate.false_alerts, 2)
        self.assertAlmostEqual(aggregate.precision, 1 / 3)
        self.assertEqual(aggregate.recall, 0.5)
        self.assertAlmostEqual(aggregate.false_alerts_per_camera_hour, 0.8)
        self.assertEqual([item.group_id for item in aggregate.by_site], ["site-a", "site-b"])
        self.assertEqual([item.group_id for item in aggregate.by_camera], ["cam-1", "cam-2"])
        self.assertEqual(
            [(item.site_id, item.camera_id) for item in aggregate.by_camera],
            [("site-a", "cam-1"), ("site-b", "cam-2")],
        )
        self.assertEqual(aggregate.median_alert_delay_ms, 3000.0)

    def test_camera_ids_are_scoped_by_site(self):
        result = evaluate_person_down_candidates([], [], video_start_timestamp_ms=0, video_end_timestamp_ms=1000)
        aggregate = aggregate_person_down_evaluations([
            EvaluationSample("east", "site-east", "cam-01", 0, 1000, result),
            EvaluationSample("west", "site-west", "cam-01", 0, 1000, result),
        ])
        self.assertEqual(aggregate.site_count, 2)
        self.assertEqual(aggregate.camera_count, 2)
        self.assertEqual(
            [(item.site_id, item.camera_id, item.group_id) for item in aggregate.by_camera],
            [("site-east", "cam-01", "cam-01"), ("site-west", "cam-01", "cam-01")],
        )

    def test_aggregate_rejects_duplicate_samples_and_overlapping_camera_time(self):
        result = evaluate_person_down_candidates([], [], video_start_timestamp_ms=0, video_end_timestamp_ms=1000)
        with self.assertRaises(ValueError):
            aggregate_person_down_evaluations([
                EvaluationSample("same", "site", "cam-a", 0, 1000, result),
                EvaluationSample("same", "site", "cam-b", 0, 1000, result),
            ])
        with self.assertRaises(ValueError):
            aggregate_person_down_evaluations([
                EvaluationSample("a", "site", "cam-a", 0, 1000, result),
                EvaluationSample("b", "site", "cam-a", 500, 1500, result),
            ])

    def test_sample_interval_must_match_result_duration(self):
        result = evaluate_person_down_candidates([], [], video_start_timestamp_ms=0, video_end_timestamp_ms=1000)
        with self.assertRaises(ValueError):
            EvaluationSample("s", "site", "cam", 0, 999, result)
        with self.assertRaises(ValueError):
            aggregate_person_down_evaluations([])


if __name__ == "__main__":
    unittest.main()
