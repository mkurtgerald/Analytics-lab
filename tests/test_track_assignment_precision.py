"""Generated association regressions; no native runtime or real media is used."""
from copy import deepcopy
from fractions import Fraction
from itertools import permutations, product
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analytics_lab.association import maximum_cardinality_weighted_matching
from analytics_lab.openvino_pipeline import run_frame_source_openvino_omz
from analytics_lab.perception import BBox, IoUTracker, TrackerConfig
from analytics_lab.video import Frame


class _GeneratedPeopleRuntime:
    runtime_version = "2026.3.1-generated"

    def __init__(self, *, departing_person, continuous_positive):
        self.departing_person = departing_person
        self.continuous_positive = continuous_positive
        self.maps = {}
        for posture in ("down", "other"):
            maps = [[[0.0] * 8 for _ in range(8)] for _ in range(19)]
            positions = (
                ((2, 2, 1), (5, 4, 1), (8, 2, 6), (11, 4, 6))
                if posture == "down" else
                ((2, 1, 2), (5, 1, 4), (8, 6, 2), (11, 6, 4))
            )
            for channel, y, x in positions:
                maps[channel][y][x] = .95
            self.maps[posture] = maps

    def detect_rows(self, image):
        # Two stationary horizontal boxes overlap at IoU=3/13, above default .15.
        rows = [(0, 1, .9, .25, .0, .65, .3)]
        if self.departing_person and image.timestamp < 2500:
            rows.insert(0, (0, 1, .9, .0, .0, .4, .3))
        return rows

    def pose_heatmaps(self, image, box):
        if box.x1 == 0:
            posture = "other" if self.continuous_positive else "down"
        else:
            posture = "down" if self.continuous_positive or image.timestamp >= 2500 else "other"
        return self.maps[posture]


class TrackAssignmentPrecisionTests(unittest.TestCase):
    def _pipeline(self, *, departing_person=True, continuous_positive=False):
        runtime = _GeneratedPeopleRuntime(
            departing_person=departing_person, continuous_positive=continuous_positive,
        )
        frames = (
            Frame(index, index * 250, SimpleNamespace(shape=(200, 400, 3), timestamp=index * 250))
            for index in range(13)
        )
        with patch("analytics_lab.openvino_omz.verify_artifact_set", return_value=()):
            return run_frame_source_openvino_omz(
                frames, artifact_root="generated-artifacts", source_id="generated-camera",
                session_id="generated-session", runtime_factory=lambda *_: runtime,
            ).video

    def test_departed_track_cannot_take_perfect_match_from_remaining_track(self):
        tracker = IoUTracker()
        departing = BBox(0, 0, 160, 60)
        remaining = BBox(100, 0, 260, 60)
        first = tracker.update(0, (departing, remaining))
        self.assertEqual(tracker.update(1, (remaining,)), (first[1],))

    def test_two_short_down_episodes_cannot_combine_into_false_candidate(self):
        control = self._pipeline(departing_person=False)
        result = self._pipeline()
        self.assertEqual(control.events, ())
        # A is down only 0..2250; B is down only 2500..3000. Neither has 3 s.
        self.assertEqual(result.events, ())
        self.assertEqual((control.observations_processed, result.observations_processed), (13, 23))

    def test_continuous_positive_survives_overlapping_person_departure(self):
        result = self._pipeline(continuous_positive=True)
        self.assertEqual(result.observations_processed, 23)
        self.assertEqual(len(result.events), 1)
        window = result.events[0]["evidence"][0]
        self.assertEqual(window["track_id"], "track-000002")
        self.assertEqual((window["start_timestamp_ms"], window["end_timestamp_ms"]), (0, 3000))

    def test_unchanged_overlapping_boxes_keep_both_handles_in_any_input_order(self):
        left, right = BBox(0, 0, 160, 60), BBox(100, 0, 260, 60)
        tracker = IoUTracker()
        first = tracker.update(0, (left, right))
        self.assertEqual(tracker.update(1, (right, left)), (first[1], first[0]))
        self.assertEqual(tracker.update(2, (left, right)), first)

    def test_existing_maximum_cardinality_augmentation_trap_is_preserved(self):
        tracker = IoUTracker(TrackerConfig(min_iou=.2))
        first = tracker.update(0, (BBox(0, 0, 10, 10), BBox(6, 0, 16, 10)))
        self.assertEqual(
            tracker.update(1, (BBox(2, 0, 12, 10), BBox(-3, 0, 7, 10))),
            (first[1], first[0]),
        )

    def test_equal_objectives_are_stable_across_detection_permutations(self):
        initial = (BBox(0, 0, 100, 40), BBox(100, 0, 200, 40))
        following = (BBox(50, -10, 150, 30), BBox(50, 10, 150, 50))
        expected = None
        for first_order, next_order in product(permutations(range(2)), repeat=2):
            tracker = IoUTracker()
            tracker.update(0, tuple(initial[index] for index in first_order))
            assigned = tracker.update(1, tuple(following[index] for index in next_order))
            canonical = tuple(assigned[next_order.index(index)] for index in range(2))
            if expected is None:
                expected = canonical
            self.assertEqual(canonical, expected)

    def test_capacity_and_assignment_failures_leave_tracker_retryable(self):
        tracker = IoUTracker(TrackerConfig(max_tracks=2))
        boxes = (BBox(0, 0, 100, 40), BBox(100, 0, 200, 40))
        first = tracker.update(0, boxes)
        before = deepcopy(tracker.__dict__)
        with self.assertRaises(RuntimeError):
            tracker.update(1, (boxes[0], BBox(300, 0, 400, 40)))
        self.assertEqual(tracker.__dict__, before)
        with patch("analytics_lab.perception.maximum_cardinality_weighted_matching", side_effect=RuntimeError("assignment failed")):
            with self.assertRaises(RuntimeError):
                tracker.update(1, boxes)
        self.assertEqual(tracker.__dict__, before)
        self.assertEqual(tracker.update(1, boxes), first)


class WeightedAssignmentTests(unittest.TestCase):
    @staticmethod
    def _oracle(weights):
        best = (0, Fraction(0))

        def enumerate_matches(row, used, count, total):
            nonlocal best
            if row == len(weights):
                best = max(best, (count, total))
                return
            enumerate_matches(row + 1, used, count, total)
            for column, weight in enumerate(weights[row]):
                if column not in used and weight is not None:
                    enumerate_matches(row + 1, used | {column}, count + 1, total + Fraction(weight))

        enumerate_matches(0, set(), 0, Fraction(0))
        return best

    def _objective(self, weights, matches):
        self.assertEqual(len({row for row, _ in matches}), len(matches))
        self.assertEqual(len({column for _, column in matches}), len(matches))
        for row, column in matches:
            self.assertIsNotNone(weights[row][column])
        return len(matches), sum((Fraction(weights[row][column]) for row, column in matches), Fraction(0))

    def test_exhaustive_tiny_matrices_match_independent_partial_matching_oracle(self):
        for rows, columns in ((1, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)):
            for values in product((None, .25, .75), repeat=rows * columns):
                weights = tuple(tuple(values[row * columns:(row + 1) * columns]) for row in range(rows))
                matches = maximum_cardinality_weighted_matching(weights, max_size=3)
                self.assertEqual(self._objective(weights, matches), self._oracle(weights), weights)

    def test_cardinality_has_priority_over_larger_total_weight(self):
        weights = ((1.0, .01, None), (None, 1.0, .01), (.01, None, None))
        self.assertEqual(
            maximum_cardinality_weighted_matching(weights, max_size=3),
            ((0, 1), (1, 2), (2, 0)),
        )

    def test_total_weight_has_priority_over_greedy_highest_edge(self):
        weights = ((.9, .8), (.89, .1))
        self.assertEqual(maximum_cardinality_weighted_matching(weights, max_size=2), ((0, 1), (1, 0)))

    def test_small_float_differences_survive_cardinality_bonus(self):
        for lower in (.5, math.ulp(0.0)):
            higher = math.nextafter(lower, 1.0)
            weights = ((lower, higher),)
            self.assertEqual(maximum_cardinality_weighted_matching(weights, max_size=2), ((0, 1),))

    def test_zero_weight_is_valid_and_none_is_absent(self):
        self.assertEqual(maximum_cardinality_weighted_matching(((None, 0.0), (0.0, None)), max_size=2),
                         ((0, 1), (1, 0)))
        self.assertEqual(maximum_cardinality_weighted_matching(((None,),), max_size=1), ())
        self.assertEqual(maximum_cardinality_weighted_matching((), max_size=1), ())
        self.assertEqual(maximum_cardinality_weighted_matching(((), ()), max_size=2), ())

    def test_invalid_dimensions_and_weights_fail_closed(self):
        for weights in (None, [], ([.5],), ((.5,), (.5, .5)),
                        ((True,),), ((float("nan"),),), ((float("inf"),),),
                        ((-.1,),), ((1.1,),), ((10 ** 1000,),)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                maximum_cardinality_weighted_matching(weights, max_size=2)
        for max_size in (False, 0, -1, 1.5):
            with self.subTest(max_size=max_size), self.assertRaises(ValueError):
                maximum_cardinality_weighted_matching((), max_size=max_size)
        for weights in (((),) * 3, ((None,) * 3,)):
            with self.assertRaises(RuntimeError):
                maximum_cardinality_weighted_matching(weights, max_size=2)

    def test_caller_capacity_can_exceed_default_tracker_size(self):
        weights = ((None,) * 256 + (1.0,),)
        self.assertEqual(maximum_cardinality_weighted_matching(weights, max_size=257), ((0, 256),))

    def test_dense_default_capacity_is_bounded_and_deterministic(self):
        weights = tuple((.5,) * 256 for _ in range(256))
        matches = maximum_cardinality_weighted_matching(weights, max_size=256)
        self.assertEqual(matches, tuple((index, index) for index in range(256)))


if __name__ == "__main__":
    unittest.main()
