import hashlib
import unittest

from analytics_lab.still_detection_evaluation import StillDetectionFrame, evaluate_still_detections
from analytics_lab.tracking import DetectionCandidate, NormalizedBox, TrackedDetection
from analytics_lab.tracking_evaluation import GroundTruthObject


BOX = NormalizedBox(.1, .1, .3, .3)
OTHER = NormalizedBox(.6, .6, .8, .8)


def label(object_id="object", box=BOX, category="vehicle"):
    return GroundTruthObject(object_id, category, box)


def detection(box=BOX, category="vehicle"):
    return DetectionCandidate(category, .9, box)


def frame(image_id, labels=(), detections=()):
    return StillDetectionFrame(image_id, hashlib.sha256(image_id.encode()).hexdigest(), labels, detections)


class StillDetectionEvaluationTests(unittest.TestCase):
    def test_mixed_matches_misses_false_detections_and_empty_exposure(self):
        result = evaluate_still_detections((
            frame("positive", (label("a"), label("b", OTHER)), (detection(),)),
            frame("negative", (), (detection(),)),
            frame("empty"),
        ))
        self.assertEqual((result.image_count, result.ground_truth_count, result.detection_count), (3, 2, 2))
        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (1, 1, 1))
        self.assertEqual((result.precision, result.recall, result.f1), (.5, .5, .5))
        self.assertAlmostEqual(result.false_detections_per_image, 1 / 3)
        self.assertEqual(result.per_image[1].false_positives, 1)
        self.assertEqual(result.per_image[2].detection_count, 0)

    def test_duplicate_detections_are_false_positives(self):
        result = evaluate_still_detections((frame("duplicate", (label(),), (detection(), detection())),))
        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (1, 1, 0))
        self.assertEqual((result.precision, result.recall), (.5, 1.))

    def test_no_predictions_counts_misses_with_undefined_precision(self):
        result = evaluate_still_detections((frame("missed", (label(),), ()),))
        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (0, 0, 1))
        self.assertIsNone(result.precision)
        self.assertEqual((result.recall, result.f1), (0., 0.))

    def test_empty_images_and_empty_evaluation_do_not_become_perfect_scores(self):
        result = evaluate_still_detections((frame("empty"),))
        self.assertEqual(result.image_count, 1)
        self.assertIsNone(result.precision)
        self.assertIsNone(result.recall)
        self.assertIsNone(result.f1)
        self.assertEqual(result.false_detections_per_image, 0.)
        empty = evaluate_still_detections(())
        self.assertEqual(empty.image_count, 0)
        self.assertIsNone(empty.false_detections_per_image)

    def test_independent_images_can_reuse_local_labels_without_cross_matching(self):
        result = evaluate_still_detections((
            frame("first", (label(),), ()), frame("second", (), (detection(),)),
            frame("third", (label(),), (detection(),)),
        ))
        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (1, 1, 1))
        self.assertFalse(hasattr(result, "id_switches"))

    def test_exact_category_and_iou_matching_use_existing_maximum_cardinality(self):
        wrong_class = evaluate_still_detections((frame("category", (label(),), (detection(category="person"),)),))
        self.assertEqual((wrong_class.true_positives, wrong_class.false_positives, wrong_class.false_negatives), (0, 1, 1))
        a, b = NormalizedBox(.1, .1, .4, .5), NormalizedBox(.16, .1, .46, .5)
        first, second = NormalizedBox(.14, .1, .44, .5), NormalizedBox(.04, .1, .34, .5)
        result = evaluate_still_detections((frame("assignment", (label("a", a), label("b", b)),
                                                (detection(first), detection(second))),))
        self.assertEqual((result.true_positives, result.false_negatives), (2, 0))

    def test_input_identity_raw_types_and_bounds_fail_closed(self):
        with self.assertRaises(ValueError):
            StillDetectionFrame("bad", "not-a-sha", (), ())
        with self.assertRaises(ValueError):
            frame("raw", (), (TrackedDetection("t", "vehicle", .9, BOX),))
        with self.assertRaises(ValueError):
            frame("labels", (label(), label()), ())
        with self.assertRaises(ValueError):
            frame("list", (), [detection()])
        duplicate_hash = StillDetectionFrame("different", frame("same").frame_sha256, (), ())
        for values in ((frame("same"), frame("same")), (frame("same"), duplicate_hash)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                evaluate_still_detections(values)
        with self.assertRaises(RuntimeError):
            evaluate_still_detections((frame("a"), frame("b")), max_images=1)
        with self.assertRaises(RuntimeError):
            evaluate_still_detections((frame("many", (), (detection(), detection())),), max_objects_per_image=1)
        for kwargs in ({"max_images": 65}, {"max_objects_per_image": 513}, {"min_iou": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_still_detections((), **kwargs)


if __name__ == "__main__":
    unittest.main()
