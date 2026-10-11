from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch

from analytics_lab.tracking import DetectionCandidate, NormalizedBox
from analytics_lab.weapons_training_manifest import WeaponTrainingAsset, WEAPON_TAXONOMY
from analytics_lab.weapons_evaluation import (
    WeaponAnnotationBox, WeaponAnnotationPackage, WeaponEvaluationCase, WeaponDonorIdentity,
    WeaponClassMetrics, evaluate_weapons, compare_weapons, main,
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def donor(name="baseline"):
    return WeaponDonorIdentity(name, "fixture/repository", "1"*40, digest(name), digest("config:"+name),
                               "fixture-runtime", "1.0", digest("runtime:"+name))


def annotation(left=0.1, right=0.4, *, object_id="object-a", category="rifle", box=None):
    return WeaponAnnotationBox(object_id, category, box or NormalizedBox(left, 0.1, right, 0.5))


def prediction(left=0.1, right=0.4, *, confidence=0.9, category="rifle", box=None):
    return DetectionCandidate(category, confidence, box or NormalizedBox(left, 0.1, right, 0.5))


def case(name="case-a", boxes=None, predictions=(), *, split="test", generated=True):
    boxes = (annotation(),) if boxes is None else boxes
    package = WeaponAnnotationPackage("annotation-"+name, digest("media:"+name), 100, 100, boxes)
    category = package.boxes[0].category if package.boxes else None
    asset = WeaponTrainingAsset(
        name, "https://example.invalid/source/"+name, "https://example.invalid/media/"+name,
        "GENERATED-METADATA", "Generated metadata fixture only", category, split, 1,
        package.media_sha256, 100, 100, package.annotation_id, package.sha256, len(package.boxes),
        (category,) if category else (), True, True, False, True,
    )
    return WeaponEvaluationCase(asset, package, predictions, generated)


class BoundProbe:
    """Length-hint access or more than bound+1 consumption is an error."""
    def __init__(self, value, maximum):
        self.value = value
        self.maximum = maximum
        self.consumed = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.consumed += 1
        if self.consumed > self.maximum + 1:
            raise AssertionError("read beyond bound+1")
        return self.value

    def __length_hint__(self):
        raise AssertionError("length hints must never be inspected")


class WeaponEvaluationTests(unittest.TestCase):
    def metrics(self, result, category="rifle"):
        return next(item for item in result.per_class if item.category == category)

    def test_per_class_and_aggregate_miss_and_negative_fp(self):
        baseline_cases = (case(), case("negative", boxes=()))
        candidate_cases = (replace(baseline_cases[0], predictions=(prediction(),)),
                           replace(baseline_cases[1], predictions=(prediction(category="handgun"),)))
        baseline = evaluate_weapons(baseline_cases, donor())
        candidate = evaluate_weapons(candidate_cases, donor("candidate"))
        self.assertEqual((baseline.true_positives, baseline.false_positives, baseline.false_negatives), (0, 0, 1))
        self.assertIsNone(baseline.precision)
        self.assertEqual(baseline.recall, 0)
        self.assertEqual((candidate.true_positives, candidate.false_positives, candidate.false_negatives), (1, 1, 0))
        self.assertEqual((candidate.precision, candidate.recall), (0.5, 1.0))
        self.assertEqual(candidate.negative_image_false_positives, 1)
        self.assertEqual(candidate.negative_images_with_false_positives, 1)
        self.assertEqual(baseline.positive_images_with_misses, 1)
        report = compare_weapons(baseline, candidate)
        self.assertEqual(report["delta_candidate_minus_baseline"]["aggregate"],
                         dict(true_positives=1, false_positives=1, false_negatives=-1, precision=None, recall=1.0))
        self.assertFalse(report["automatic_promotion"])
        self.assertFalse(report["accuracy_claim"])
        json.dumps(report, allow_nan=False)

    def test_maximum_cardinality_augments_instead_of_confidence_greedy(self):
        boxes = (annotation(0.1, 0.4), annotation(0.16, 0.46, object_id="object-b"))
        predictions = (prediction(0.12, 0.42, confidence=0.95), prediction(0.04, 0.34, confidence=0.6))
        result = evaluate_weapons((case(boxes=boxes, predictions=predictions),), donor())
        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (2, 0, 0))
        reversed_result = evaluate_weapons((case(boxes=tuple(reversed(boxes)), predictions=tuple(reversed(predictions))),), donor())
        self.assertEqual(result, reversed_result)
        self.assertEqual(result.to_dict()["protocol"]["matching"], "deterministic_one_to_one_maximum_cardinality")

    def test_three_object_augmentation_chain(self):
        boxes = (annotation(0.1, 0.4), annotation(0.2, 0.5, object_id="object-b"),
                 annotation(0.205, 0.505, object_id="object-c"))
        predictions = (prediction(0.14, 0.44), prediction(0.04, 0.34), prediction(0.26, 0.56))
        result = evaluate_weapons((case(boxes=boxes, predictions=predictions),), donor())
        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (3, 0, 0))

    def test_class_exact_matching_including_knife_subclasses(self):
        for actual, predicted in (("rifle", "weapon"), ("kitchen_knife", "knife")):
            with self.subTest(actual=actual, predicted=predicted):
                result = evaluate_weapons((case(boxes=(annotation(category=actual),),
                                                predictions=(prediction(category=predicted),)),), donor())
                self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (0, 1, 1))
                self.assertEqual(result.duplicate_predictions, 0)

    def test_iou_and_score_thresholds_are_inclusive_and_fixed(self):
        full = NormalizedBox(0, 0, 1, 1)
        for confidence, right, matched in ((0.5, 0.5, True), (0.5, 0.499999, False), (0.499999, 1, False)):
            with self.subTest(confidence=confidence, right=right):
                result = evaluate_weapons((case(boxes=(annotation(box=full),), predictions=(
                    prediction(box=NormalizedBox(0, 0, right, 1), confidence=confidence),)),), donor())
                self.assertEqual(result.true_positives, int(matched))
                self.assertEqual(result.false_negatives, int(not matched))
                self.assertEqual(result.false_positives, int(confidence >= 0.5 and not matched))

    def test_tiny_normalized_boxes_do_not_underflow_iou(self):
        for tiny in (1e-200, 5e-324):
            with self.subTest(tiny=tiny):
                box = NormalizedBox(0, 0, tiny, tiny)
                result = evaluate_weapons((case(boxes=(annotation(box=box),), predictions=(prediction(box=box),)),), donor())
                self.assertEqual(result.true_positives, 1)

    def test_duplicates_are_unmatched_predictions_with_matching_gt_edges(self):
        result = evaluate_weapons((case(predictions=(prediction(), prediction(), prediction(0.7, 0.9))),), donor())
        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (1, 2, 0))
        self.assertEqual(result.duplicate_predictions, 1)

    def test_true_negative_and_undefined_denominators(self):
        clean = evaluate_weapons((case(boxes=()),), donor())
        self.assertEqual(clean.negative_images, 1)
        self.assertEqual(clean.negative_image_false_positives, 0)
        self.assertIsNone(clean.precision)
        self.assertIsNone(clean.recall)
        for item in clean.per_class:
            self.assertIsNone(item.precision)
            self.assertIsNone(item.recall)
        alerts = evaluate_weapons((case(boxes=(), predictions=(prediction(), prediction())),), donor())
        self.assertEqual(alerts.negative_image_false_positives, 2)
        self.assertEqual(alerts.negative_images_with_false_positives, 1)
        self.assertEqual(alerts.duplicate_predictions, 0)

    def test_annotation_canonical_hash_binds_boxes_and_order(self):
        a, b = annotation(), annotation(0.16, 0.46, object_id="object-b")
        first = WeaponAnnotationPackage("annotation", digest("media"), 100, 100, (a, b))
        second = replace(first, boxes=(b, a))
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.sha256, hashlib.sha256(first.canonical_bytes()).hexdigest())
        self.assertNotEqual(first.sha256, replace(first, boxes=(replace(a, box=NormalizedBox(0, 0, 1, 1)), b)).sha256)
        with self.assertRaises(ValueError):
            replace(first, boxes=(a, a))

    def test_annotation_and_asset_binding_fail_closed(self):
        original = case()
        changes = (dict(annotation_id="other"), dict(media_sha256=digest("other-media")),
                   dict(width=101), dict(height=101), dict(boxes=()))
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(original, annotations=replace(original.annotations, **change))
        for asset_change in (dict(annotation_sha256="0"*64), dict(annotation_box_count=2)):
            with self.subTest(asset_change=asset_change), self.assertRaises(ValueError):
                replace(original, asset=replace(original.asset, **asset_change))
        mismatched_classes = replace(original.annotations, boxes=(annotation(category="knife"),))
        changed_asset = replace(original.asset, annotation_sha256=mismatched_classes.sha256)
        with self.assertRaises(ValueError):
            replace(original, asset=changed_asset, annotations=mismatched_classes)

    def test_train_and_uncleared_rights_rejected(self):
        original = case()
        for changes in (dict(split="train"), dict(commercial_evaluation_allowed=False),
                        dict(identifiable_people=True), dict(third_party_rights_cleared=False)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(original, asset=replace(original.asset, **changes))
        self.assertEqual(case(split="validation").asset.split, "validation")

    def test_defensive_copies_and_revalidation_of_mutated_records(self):
        original_box = NormalizedBox(0.1, 0.1, 0.4, 0.5)
        original_prediction = prediction(box=original_box)
        original_annotation = annotation(box=original_box)
        original = case(boxes=(original_annotation,), predictions=[original_prediction])
        object.__setattr__(original_box, "x_max", 0.9)
        object.__setattr__(original_prediction, "confidence", 0.1)
        object.__setattr__(original_annotation, "category", "knife")
        self.assertEqual(original.predictions[0].confidence, 0.9)
        self.assertEqual(original.predictions[0].box.x_max, 0.4)
        self.assertEqual(original.annotations.boxes[0].category, "rifle")
        with self.assertRaises(FrozenInstanceError):
            original.generated_metadata = False
        for field, value in (("confidence", float("nan")), ("category", "vehicle")):
            fresh = case(predictions=(prediction(),))
            object.__setattr__(fresh.predictions[0], field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_weapons((fresh,), donor())
        fresh = case()
        object.__setattr__(fresh.asset, "commercial_evaluation_allowed", False)
        with self.assertRaises(ValueError):
            evaluate_weapons((fresh,), donor())

    def test_bound_plus_one_without_length_hints(self):
        original = case()
        for name, values, action in (
            ("predictions", BoundProbe(prediction(), 64), lambda values: replace(original, predictions=values)),
            ("annotations", BoundProbe(annotation(), 64), lambda values: replace(original.annotations, boxes=values)),
            ("cases", BoundProbe(original, 256), lambda values: evaluate_weapons(values, donor())),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                action(values)
            self.assertEqual(values.consumed, values.maximum + 1)

    def test_exact_64_and_256_bounds_are_accepted(self):
        predictions = tuple(prediction() for _ in range(64))
        annotations = tuple(annotation(object_id=f"object-{index}") for index in range(64))
        result = evaluate_weapons((case(boxes=annotations, predictions=predictions),), donor())
        self.assertEqual(result.true_positives, 64)
        cases = (case(f"case-{index}", boxes=()) for index in range(256))
        result = evaluate_weapons(cases, donor())
        self.assertEqual((result.case_count, result.negative_images), (256, 256))

    def test_asset_sizes_urls_and_mutated_class_summary_are_bounded(self):
        original = case()
        for changes in (dict(expected_size=2**63), dict(source_page_url="https://example.invalid/"+"x"*2048)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(original, asset=replace(original.asset, **changes))
        object.__setattr__(original.asset, "annotation_classes", ["rifle"])
        with self.assertRaises(ValueError):
            evaluate_weapons((original,), donor())

    def test_dataset_digest_includes_rights_annotations_excludes_predictions(self):
        original = case()
        baseline = evaluate_weapons((original,), donor())
        predicted = evaluate_weapons((replace(original, predictions=(prediction(),)),), donor("other"))
        self.assertEqual(baseline.dataset_sha256, predicted.dataset_sha256)
        changed_rights = replace(original, asset=replace(original.asset, rights_basis="Another asserted rights basis"))
        changed = evaluate_weapons((changed_rights,), donor())
        self.assertNotEqual(baseline.dataset_sha256, changed.dataset_sha256)
        with self.assertRaises(ValueError):
            compare_weapons(baseline, changed)
        changed_package = replace(original.annotations, boxes=(annotation(0.15, 0.45),))
        changed = replace(original, annotations=changed_package,
                          asset=replace(original.asset, annotation_sha256=changed_package.sha256))
        self.assertNotEqual(baseline.dataset_sha256, evaluate_weapons((changed,), donor()).dataset_sha256)

    def test_case_order_deterministic_and_duplicate_identity_rejected(self):
        first, second = case(), case("negative", boxes=())
        self.assertEqual(evaluate_weapons((first, second), donor()), evaluate_weapons((second, first), donor()))
        with self.assertRaises(ValueError):
            evaluate_weapons((first, first), donor())
        for field in ("asset_id", "sha256", "annotation_id"):
            asset = replace(second.asset, **{field: getattr(first.asset, field)})
            package = second.annotations
            if field == "sha256":
                package = replace(package, media_sha256=first.asset.sha256)
            if field == "annotation_id":
                package = replace(package, annotation_id=first.asset.annotation_id)
            asset = replace(asset, annotation_sha256=package.sha256)
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_weapons((first, replace(second, asset=asset, annotations=package)), donor())

    def test_comparison_revalidates_protocol_results_and_donor_identity(self):
        baseline = evaluate_weapons((case(),), donor())
        different_model = evaluate_weapons((case(predictions=(prediction(),)),), donor("new-model"))
        self.assertFalse(compare_weapons(baseline, different_model)["automatic_promotion"])
        for field, value in (("protocol_sha256", digest("other-protocol")), ("negative_image_false_positives", 1),
                             ("generated_cases", 0)):
            forged = replace(baseline)
            object.__setattr__(forged, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_weapons(baseline, forged)
        changed = replace(baseline)
        object.__setattr__(changed.donor, "model_sha256", "x"*64)
        with self.assertRaises(ValueError):
            compare_weapons(baseline, changed)

    def test_mixed_generated_metadata_is_explicit(self):
        result = evaluate_weapons((case(), case("other", boxes=(), generated=False)), donor())
        self.assertFalse(result.generated_metadata)
        self.assertEqual(result.generated_cases, 1)
        self.assertTrue(result.to_dict()["mixed_generated_and_other_metadata"])
        self.assertFalse(result.to_dict()["accuracy_claim"])

    def test_impossible_positive_negative_and_duplicate_counts_rejected(self):
        negative = evaluate_weapons((case(boxes=()),), donor())
        for tp, fn in ((1, 0), (0, 1)):
            metrics = tuple(WeaponClassMetrics(category, tp if category == "rifle" else 0, 0,
                                               fn if category == "rifle" else 0) for category in WEAPON_TAXONOMY)
            with self.subTest(tp=tp, fn=fn), self.assertRaises(ValueError):
                replace(negative, per_class=metrics)
        with self.assertRaises(ValueError):
            replace(negative, negative_images=0)
        alert = evaluate_weapons((case(boxes=(), predictions=(prediction(),)),), donor())
        with self.assertRaises(ValueError):
            replace(alert, duplicate_predictions=1)
        missed = evaluate_weapons((case(), case("second-positive")), donor())
        with self.assertRaises(ValueError):
            replace(missed, positive_images_with_misses=1)
        wrong_class = evaluate_weapons((case(predictions=(prediction(category="knife"),)),), donor())
        with self.assertRaises(ValueError):
            replace(wrong_class, duplicate_predictions=1)
        bounded_duplicates = evaluate_weapons((case(predictions=tuple(prediction() for _ in range(64))),
                                                case("second-positive", predictions=(prediction(category="weapon"),))), donor())
        self.assertEqual(bounded_duplicates.duplicate_predictions, 63)
        with self.assertRaises(ValueError):
            replace(bounded_duplicates, duplicate_predictions=64)

    def test_bad_types_empty_runs_and_unknown_categories(self):
        for values in ((), (object(),), "not cases"):
            with self.subTest(values=values), self.assertRaises(ValueError):
                evaluate_weapons(values, donor())
        with self.assertRaises(ValueError):
            case(predictions=(DetectionCandidate("person", 0.9, NormalizedBox(0, 0, 1, 1)),))
        with self.assertRaises(ValueError):
            WeaponAnnotationPackage("a", "F"*64, 100, 100, ())
        with self.assertRaises(ValueError):
            WeaponClassMetrics("rifle", True, 0, 0)
        for changes in (dict(code_revision="x"*40), dict(runtime_sha256="0"*63), dict(runtime_name="\n")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(donor(), **changes)

    def test_synthetic_demo_cli_has_no_accuracy_or_readiness_claim(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["weapons_evaluation", "--synthetic-demo"]), redirect_stdout(output):
            self.assertEqual(main(), 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report["generated_metadata"])
        self.assertEqual(report["baseline"]["aggregate"]["false_negatives"], 1)
        self.assertEqual(report["candidate"]["aggregate"]["true_positives"], 1)
        self.assertEqual(report["candidate"]["negative_image_false_positives"], 1)
        self.assertFalse(report["accuracy_claim"])
        self.assertFalse(report["readiness_claim"])
        self.assertEqual(report["candidate"]["donor"]["runtime_name"], "no-inference")


if __name__ == "__main__":
    unittest.main()
