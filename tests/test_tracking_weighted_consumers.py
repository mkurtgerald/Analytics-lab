"""Generated detector/association/evaluation regressions; no native runtime."""
from itertools import permutations
import unittest
from dataclasses import asdict
from unittest.mock import patch

from analytics_lab.association import maximum_cardinality_weighted_matching
from analytics_lab.bytetrack import ByteTrackAssociationBackend, ByteTrackAssociationConfig
from analytics_lab.openvino_omz import parse_person_detections
from analytics_lab.tracking import (
    DetectionCandidate, NormalizedBox, TrackedDetection, TrackingSession,
)
from analytics_lab.tracking_evaluation import (
    GroundTruthObject, TrackingEvaluationConfig, TrackingEvaluationFrame,
    _match_frame, evaluate_tracking,
)


def detections(rows, label=1):
    """Use the actual reviewed SSD parser and normalize its pixel boxes."""
    return tuple(
        DetectionCandidate("person", item.confidence, NormalizedBox(
            item.bbox.x1 / 100, item.bbox.y1 / 100,
            item.bbox.x2 / 100, item.bbox.y2 / 100,
        ), label)
        for item in parse_person_detections(
            rows, image_width=100, image_height=100,
            threshold=.50, max_people=16, person_label=label,
        )
    )


class WeightedTrackingConsumerTests(unittest.TestCase):
    def test_actual_ssd_to_bytetrack_preserves_stationary_remaining_person(self):
        for label in (0, 1):
            a = (0, label, .9, .10, .10, .50, .50)
            b = (0, label, .9, .35, .10, .75, .50)
            for initial_rows in permutations((a, b)):
                for remaining in ((a,), (b,), (a, b), (b, a)):
                    with self.subTest(label=label, remaining=remaining):
                        session = TrackingSession(ByteTrackAssociationBackend())
                        first = session.update(0, 0, detections(initial_rows, label))
                        second = session.update(1, 33, detections(remaining, label))
                        original_ids = {item.box: item.track_id for item in first}
                        self.assertEqual(
                            {item.box: item.track_id for item in second},
                            {item.box: original_ids[item.box] for item in second},
                        )
                        self.assertEqual(len(second), len(remaining))
                        truth = {item.box: f"object-{index}" for index, item in enumerate(first)}
                        frames = tuple(
                            TrackingEvaluationFrame(index, tuple(
                                GroundTruthObject(truth[item.box], "person", item.box)
                                for item in output
                            ), output)
                            for index, output in enumerate((first, second))
                        )
                        result = evaluate_tracking(frames)
                        self.assertEqual((result.misses, result.false_track_observations,
                                          result.id_switches), (0, 0, 0))

    def test_low_and_unconfirmed_stages_keep_the_best_remaining_link(self):
        a = DetectionCandidate("person", .9, NormalizedBox(.1, .1, .5, .5), 0)
        b = DetectionCandidate("person", .9, NormalizedBox(.2, .1, .6, .5), 0)
        for stage in ("low", "unconfirmed"):
            with self.subTest(stage=stage):
                backend = ByteTrackAssociationBackend()
                session = TrackingSession(backend)
                if stage == "unconfirmed":
                    session.update(0, 0, ())
                    self.assertEqual(session.update(1, 33, (a, b)), ())
                    expected = next(item.track_id for item in backend._tracks.values() if item.box == b.box)
                    output = session.update(2, 66, (b,))
                else:
                    first = session.update(0, 0, (a, b))
                    expected = next(item.track_id for item in first if item.box == b.box)
                    low = DetectionCandidate("person", .2, b.box, 0)
                    output = session.update(1, 33, (low,))
                self.assertEqual([(item.track_id, item.box) for item in output], [(expected, b.box)])

    def test_direct_detector_order_does_not_change_birth_or_retained_ids(self):
        initial = tuple(DetectionCandidate("person", .9, NormalizedBox(x, .1, x + .2, .6), 0)
                        for x in (.1, .25, .7))
        expected = None
        for first_order in permutations(initial):
            for second_order in permutations(initial[1:]):
                session = TrackingSession(ByteTrackAssociationBackend())
                result = tuple(
                    {item.box: item.track_id for item in output}
                    for output in (session.update(0, 0, first_order), session.update(1, 33, second_order))
                )
                if expected is None:
                    expected = result
                self.assertEqual(result, expected)
                self.assertEqual(result[1], {item.box: result[0][item.box] for item in initial[1:]})

    def test_bytetrack_preserves_maximum_valid_link_count(self):
        def detection(left):
            return DetectionCandidate("person", .9, NormalizedBox(left, .1, left + .2, .5), 0)
        a, b = detection(.1), detection(.24)
        common, exclusive = detection(.17), detection(.03)
        session = TrackingSession(ByteTrackAssociationBackend())
        first = session.update(0, 0, (a, b))
        ids = {item.box: item.track_id for item in first}
        second = session.update(1, 33, (common, exclusive))
        self.assertEqual({item.box: item.track_id for item in second},
                         {exclusive.box: ids[a.box], common.box: ids[b.box]})

    def test_weighted_stage_failure_is_atomic_and_uses_configured_capacity(self):
        backend = ByteTrackAssociationBackend(ByteTrackAssociationConfig(max_tracks=2))
        session = TrackingSession(backend)
        detection = DetectionCandidate("person", .9, NormalizedBox(.1, .1, .3, .6), 0)
        first = session.update(0, 0, (detection,))
        def snapshot():
            return ({key: asdict(value) for key, value in backend._tracks.items()},
                    backend._next_id, backend._first_frame, backend._last_frame,
                    session._last_frame, session._last_timestamp)
        before = snapshot()
        calls = []
        def failing_matcher(weights, *, max_size):
            calls.append(max_size)
            if len(calls) == 2:
                raise RuntimeError("generated second-stage failure")
            return maximum_cardinality_weighted_matching(weights, max_size=max_size)
        with patch("analytics_lab.bytetrack.maximum_cardinality_weighted_matching", side_effect=failing_matcher):
            with self.assertRaisesRegex(RuntimeError, "second-stage failure"):
                session.update(1, 33, (detection,))
        self.assertEqual(calls, [2, 2])
        self.assertEqual(snapshot(), before)
        self.assertEqual(session.update(1, 33, (detection,)), first)

    def test_evaluator_overlap_jitter_keeps_oracle_ids_and_counts_real_switches(self):
        a = NormalizedBox(.10, .10, .40, .50)
        b = NormalizedBox(.14, .10, .44, .50)
        jitter_b = NormalizedBox(.05, .10, .35, .50)
        truth = (GroundTruthObject("A", "person", a), GroundTruthObject("B", "person", b))
        first = (TrackedDetection("track-A", "person", .9, a),
                 TrackedDetection("track-B", "person", .9, b))
        for exchanged, expected_switches in ((False, 0), (True, 2)):
            second = (TrackedDetection("track-B" if exchanged else "track-A", "person", .9, a),
                      TrackedDetection("track-A" if exchanged else "track-B", "person", .9, jitter_b))
            for truth_order in permutations(truth):
                for output_order in permutations(second):
                    with self.subTest(exchanged=exchanged, order=output_order):
                        result = evaluate_tracking((
                            TrackingEvaluationFrame(0, truth_order, first),
                            TrackingEvaluationFrame(1, truth_order, output_order),
                        ))
                        self.assertEqual(result.id_switches, expected_switches)
                        self.assertEqual((result.matched_observations, result.misses,
                                          result.false_track_observations), (4, 0, 0))
                        # Three exact matches plus the independent interval IoU 7/13.
                        self.assertAlmostEqual(result.mean_matched_iou, 23 / 26)

    def test_evaluator_equal_objectives_use_canonical_ids(self):
        box = NormalizedBox(.1, .1, .3, .5)
        truth = (GroundTruthObject("B", "person", box), GroundTruthObject("A", "person", box))
        outputs = (TrackedDetection("z", "person", .9, box), TrackedDetection("a", "person", .9, box))
        expected = None
        for gt_order in permutations(truth):
            for track_order in permutations(outputs):
                pairs, missing, extra = _match_frame(gt_order, track_order, .5)
                identities = {(gt_order[g].object_id, track_order[t].track_id) for g, t, _ in pairs}
                if expected is None:
                    expected = identities
                self.assertEqual(identities, expected)
                self.assertEqual((missing, extra), ((), ()))

    def test_evaluator_passes_custom_bound_and_rejects_overflow_before_matching(self):
        box = NormalizedBox(.1, .1, .3, .5)
        truth = (GroundTruthObject("A", "person", box),)
        outputs = tuple(TrackedDetection(f"vehicle-{i}", "vehicle", .9, box) for i in range(512))
        outputs += (TrackedDetection("person", "person", .9, box),)
        frame = TrackingEvaluationFrame(0, truth, outputs)
        result = evaluate_tracking((frame,), TrackingEvaluationConfig(max_objects_per_frame=513))
        self.assertEqual((result.matched_observations, result.misses,
                          result.false_track_observations), (1, 0, 512))
        with patch("analytics_lab.tracking_evaluation.maximum_cardinality_weighted_matching") as matcher:
            with self.assertRaisesRegex(RuntimeError, "object count exceeds"):
                evaluate_tracking((frame,), TrackingEvaluationConfig(max_objects_per_frame=512))
            matcher.assert_not_called()


if __name__ == "__main__":
    unittest.main()
