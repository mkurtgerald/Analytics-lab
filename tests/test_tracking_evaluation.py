import unittest

from analytics_lab.tracking import NormalizedBox, TrackedDetection
from analytics_lab.tracking_evaluation import (
    GroundTruthObject,
    TrackingEvaluationConfig,
    TrackingEvaluationFrame,
    evaluate_tracking,
)


def box(x1, y1, x2, y2):
    return NormalizedBox(x1, y1, x2, y2)


def gt(object_id, x1, y1, x2, y2, category="person"):
    return GroundTruthObject(object_id, category, box(x1, y1, x2, y2))


def track(track_id, x1, y1, x2, y2, category="person"):
    return TrackedDetection(track_id, category, 0.9, box(x1, y1, x2, y2))


class TrackingEvaluationTests(unittest.TestCase):
    def test_perfect_continuity(self):
        frames = (
            TrackingEvaluationFrame(0, (gt("p1", .1, .1, .3, .5),), (track("t1", .1, .1, .3, .5),)),
            TrackingEvaluationFrame(1, (gt("p1", .2, .1, .4, .5),), (track("t1", .2, .1, .4, .5),)),
        )
        result = evaluate_tracking(frames)
        self.assertEqual(result.ground_truth_observations, 2)
        self.assertEqual(result.matched_observations, 2)
        self.assertEqual(result.misses, 0)
        self.assertEqual(result.false_track_observations, 0)
        self.assertEqual(result.id_switches, 0)
        self.assertEqual(result.fragmentations, 0)
        self.assertEqual(result.mean_matched_iou, 1.0)
        self.assertEqual(result.continuity, 1.0)

    def test_direct_id_switch_without_fragmentation(self):
        frames = (
            TrackingEvaluationFrame(0, (gt("p1", .1, .1, .3, .5),), (track("t1", .1, .1, .3, .5),)),
            TrackingEvaluationFrame(1, (gt("p1", .1, .1, .3, .5),), (track("t2", .1, .1, .3, .5),)),
        )
        result = evaluate_tracking(frames)
        self.assertEqual(result.id_switches, 1)
        self.assertEqual(result.fragmentations, 0)

    def test_gap_then_reacquisition_counts_fragmentation(self):
        frames = (
            TrackingEvaluationFrame(0, (gt("p1", .1, .1, .3, .5),), (track("t1", .1, .1, .3, .5),)),
            TrackingEvaluationFrame(1, (gt("p1", .1, .1, .3, .5),), ()),
            TrackingEvaluationFrame(2, (gt("p1", .1, .1, .3, .5),), (track("t1", .1, .1, .3, .5),)),
        )
        result = evaluate_tracking(frames)
        self.assertEqual(result.misses, 1)
        self.assertEqual(result.fragmentations, 1)
        self.assertEqual(result.id_switches, 0)

    def test_gap_then_new_id_counts_fragmentation_and_switch(self):
        frames = (
            TrackingEvaluationFrame(0, (gt("p1", .1, .1, .3, .5),), (track("t1", .1, .1, .3, .5),)),
            TrackingEvaluationFrame(1, (gt("p1", .1, .1, .3, .5),), ()),
            TrackingEvaluationFrame(2, (gt("p1", .1, .1, .3, .5),), (track("t2", .1, .1, .3, .5),)),
        )
        result = evaluate_tracking(frames)
        self.assertEqual(result.fragmentations, 1)
        self.assertEqual(result.id_switches, 1)

    def test_false_track_and_category_mismatch_are_counted(self):
        frame = TrackingEvaluationFrame(
            0,
            (gt("p1", .1, .1, .3, .5),),
            (
                track("vehicle-track", .1, .1, .3, .5, category="vehicle"),
                track("extra", .6, .1, .8, .5),
            ),
        )
        result = evaluate_tracking((frame,))
        self.assertEqual(result.misses, 1)
        self.assertEqual(result.false_track_observations, 2)
        self.assertEqual(result.matched_observations, 0)

    def test_matching_preserves_maximum_cardinality(self):
        # p1 can match either track, p2 can only match t1. A naive greedy match
        # can consume t1 for p1 and miss p2; augmenting-path matching keeps both.
        frames = (
            TrackingEvaluationFrame(
                0,
                (
                    gt("p1", .10, .10, .40, .50),
                    gt("p2", .16, .10, .46, .50),
                ),
                (
                    track("t1", .14, .10, .44, .50),
                    track("t2", .04, .10, .34, .50),
                ),
            ),
        )
        result = evaluate_tracking(frames, TrackingEvaluationConfig(min_iou=.5))
        self.assertEqual(result.matched_observations, 2)
        self.assertEqual(result.misses, 0)
        self.assertEqual(result.false_track_observations, 0)

    def test_bounds_and_order_fail_closed(self):
        with self.assertRaises(ValueError):
            TrackingEvaluationFrame(-1, (), ())
        with self.assertRaises(ValueError):
            evaluate_tracking((
                TrackingEvaluationFrame(1, (), ()),
                TrackingEvaluationFrame(1, (), ()),
            ))
        with self.assertRaises(ValueError):
            TrackingEvaluationConfig(max_objects_per_frame=0)
        with self.assertRaises(RuntimeError):
            evaluate_tracking(
                (
                    TrackingEvaluationFrame(
                        0,
                        (
                            gt("p1", .1, .1, .3, .5),
                            gt("p2", .5, .1, .7, .5),
                        ),
                        (),
                    ),
                ),
                TrackingEvaluationConfig(max_objects_per_frame=1),
            )


if __name__ == "__main__":
    unittest.main()
