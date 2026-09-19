import unittest

from analytics_lab.tracking import NormalizedBox
from analytics_lab.tracking_annotations import (
    TrackingGroundTruthFrame,
    canonical_tracking_annotations_bytes,
    canonical_tracking_annotations_sha256,
)
from analytics_lab.tracking_benchmark_plan import wikimedia_pedestrian_smoke_plan
from analytics_lab.tracking_evaluation import GroundTruthObject


def gt(object_id, box=(.1, .1, .2, .4), category="person"):
    return GroundTruthObject(object_id, category, NormalizedBox(*box))


def full_window(first_objects=()):
    plan = wikimedia_pedestrian_smoke_plan()
    return tuple(
        TrackingGroundTruthFrame(index, tuple(first_objects) if index == plan.start_frame else ())
        for index in plan.frame_indices
    )


class TrackingAnnotationBindingTests(unittest.TestCase):
    def test_complete_window_is_bound_to_plan_and_order_independent_objects(self):
        plan = wikimedia_pedestrian_smoke_plan()
        a = gt("person-b", (.3, .1, .4, .5))
        b = gt("person-a", (.1, .1, .2, .5))
        first = full_window((a, b))
        second = full_window((b, a))
        self.assertEqual(
            canonical_tracking_annotations_sha256(plan, first),
            canonical_tracking_annotations_sha256(plan, second),
        )
        payload = canonical_tracking_annotations_bytes(plan, first)
        self.assertIn(plan.canonical_sha256().encode("ascii"), payload)
        self.assertIn(plan.source_sha256.encode("ascii"), payload)

    def test_missing_or_reordered_frame_fails_closed(self):
        plan = wikimedia_pedestrian_smoke_plan()
        values = full_window()
        with self.assertRaises(ValueError):
            canonical_tracking_annotations_sha256(plan, values[:-1])
        with self.assertRaises(ValueError):
            canonical_tracking_annotations_sha256(plan, tuple(reversed(values)))

    def test_wrong_class_and_duplicate_object_id_fail_closed(self):
        plan = wikimedia_pedestrian_smoke_plan()
        with self.assertRaises(ValueError):
            canonical_tracking_annotations_sha256(plan, full_window((gt("v1", category="vehicle"),)))
        with self.assertRaises(ValueError):
            TrackingGroundTruthFrame(plan.start_frame, (gt("p1"), gt("p1", (.3, .1, .4, .5))))

    def test_empty_frames_are_explicit_and_bounded(self):
        plan = wikimedia_pedestrian_smoke_plan()
        values = full_window()
        digest = canonical_tracking_annotations_sha256(plan, values)
        self.assertEqual(len(digest), 64)
        with self.assertRaises(RuntimeError):
            canonical_tracking_annotations_bytes(plan, values, max_canonical_bytes=1)
        crowded = full_window((gt("p1"), gt("p2", (.3, .1, .4, .5))))
        with self.assertRaises(RuntimeError):
            canonical_tracking_annotations_bytes(plan, crowded, max_objects_per_frame=1)

    def test_frame_container_rejects_non_ground_truth_values(self):
        plan = wikimedia_pedestrian_smoke_plan()
        with self.assertRaises(ValueError):
            TrackingGroundTruthFrame(plan.start_frame, (object(),))


if __name__ == "__main__":
    unittest.main()
