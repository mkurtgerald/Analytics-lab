"""Bounded donor-neutral Weapons annotation, evaluation and comparison contracts.

This module processes metadata only, verifies canonical annotation bytes against
the asset declaration, and never opens media, models or a network connection.
Asset rights, annotation authorship and media association remain caller supplied;
hash equality is not authenticity. Counts use a distinct maximum-cardinality
protocol, not AP/COCO, and never confer accuracy, readiness or action authority.
Each case admits at most 64 predictions and 64 annotation boxes; a run admits
1-256 held-out cases. Collectors consume at most one item beyond each bound.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import argparse
import hashlib
import json
import re
from typing import Iterable

from .tracking import DetectionCandidate, NormalizedBox
from .weapons_training_manifest import WEAPON_TAXONOMY, WeaponTrainingAsset

_MAX_ITEMS = 64
_MAX_CASES = 256
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_PROTOCOL = {
    "schema": "analytics.weapons-evaluation.v1", "confidence_cutoff": 0.5,
    "iou_threshold": 0.5, "class_matching": "exact_frozen_seven_class_taxonomy",
    "matching": "deterministic_one_to_one_maximum_cardinality",
    "duplicate": "unmatched_accepted_prediction_with_class_exact_gt_edge",
    "taxonomy": WEAPON_TAXONOMY,
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


PROTOCOL_SHA256 = _hash(_PROTOCOL)


def _bounded(values: Iterable, limit: int, name: str) -> tuple:
    result = []
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be iterable") from exc
    for value in iterator:
        if len(result) == limit:
            raise ValueError(f"{name} exceeds its {limit} item bound")
        result.append(value)
    return tuple(result)


def _text(value: object, name: str, maximum: int = 512) -> None:
    if (type(value) is not str or not value.strip() or len(value) > maximum
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)):
        raise ValueError(f"{name} must be bounded nonempty text")


def _digest(value: object, name: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} requires lowercase SHA-256")


def _integer(value: object, name: str, maximum: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer within [{minimum}, {maximum}]")


def _box(value: NormalizedBox) -> NormalizedBox:
    if type(value) is not NormalizedBox:
        raise ValueError("box must be a NormalizedBox")
    return replace(value)


def _iou(a: NormalizedBox, b: NormalizedBox) -> float:
    """Scale each axis before area products so tiny valid boxes cannot underflow."""
    aw, ah = a.x_max - a.x_min, a.y_max - a.y_min
    bw, bh = b.x_max - b.x_min, b.y_max - b.y_min
    iw = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    ih = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if iw <= 0 or ih <= 0:
        return 0.0
    sx, sy = max(aw, bw), max(ah, bh)
    intersection = (iw / sx) * (ih / sy)
    union = (aw / sx) * (ah / sy) + (bw / sx) * (bh / sy) - intersection
    return intersection / union


@dataclass(frozen=True)
class WeaponAnnotationBox:
    object_id: str
    category: str
    box: NormalizedBox

    def __post_init__(self) -> None:
        _text(self.object_id, "object_id", 128)
        if type(self.category) is not str or self.category not in WEAPON_TAXONOMY:
            raise ValueError("annotation category is outside the frozen taxonomy")
        object.__setattr__(self, "box", _box(self.box))


@dataclass(frozen=True)
class WeaponAnnotationPackage:
    annotation_id: str
    media_sha256: str
    width: int
    height: int
    boxes: tuple[WeaponAnnotationBox, ...]

    def __post_init__(self) -> None:
        _text(self.annotation_id, "annotation_id")
        _digest(self.media_sha256, "media_sha256")
        _integer(self.width, "width", 32768, 1)
        _integer(self.height, "height", 32768, 1)
        boxes = _bounded(self.boxes, _MAX_ITEMS, "annotation boxes")
        if any(type(item) is not WeaponAnnotationBox for item in boxes):
            raise ValueError("annotation boxes contain unsupported records")
        copied = tuple(replace(item) for item in boxes)
        if len({item.object_id for item in copied}) != len(copied):
            raise ValueError("duplicate annotation object_id")
        object.__setattr__(self, "boxes", tuple(sorted(copied, key=lambda item: item.object_id)))

    def canonical_bytes(self) -> bytes:
        return _canonical({"schema": "analytics.weapons-annotations.v1", **asdict(replace(self))})

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class WeaponEvaluationCase:
    asset: WeaponTrainingAsset
    annotations: WeaponAnnotationPackage
    predictions: tuple[DetectionCandidate, ...]
    generated_metadata: bool = False

    def __post_init__(self) -> None:
        if type(self.asset) is not WeaponTrainingAsset or type(self.annotations) is not WeaponAnnotationPackage:
            raise ValueError("case requires typed asset and annotation records")
        _integer(self.asset.expected_size, "expected_size", 2**63 - 1, 1)
        if type(self.asset.annotation_classes) is not tuple or len(self.asset.annotation_classes) > 1:
            raise ValueError("asset class summary must be an immutable zero/one-class tuple")
        for name in ("source_page_url", "asset_url"):
            _text(getattr(self.asset, name), name, 2048)
        asset, annotations = replace(self.asset), replace(self.annotations)
        if asset.split not in ("validation", "test") or not asset.eligible_for_declared_split():
            raise ValueError("evaluation requires rights-cleared held-out assets without identifiable people")
        if (annotations.annotation_id != asset.annotation_id or annotations.sha256 != asset.annotation_sha256
                or annotations.media_sha256 != asset.sha256
                or (annotations.width, annotations.height) != (asset.width, asset.height)):
            raise ValueError("annotation identity/hash/media/dimensions do not bind the asset")
        classes = tuple(sorted({item.category for item in annotations.boxes}))
        if len(annotations.boxes) != asset.annotation_box_count or classes != asset.annotation_classes:
            raise ValueError("actual annotation boxes/classes disagree with the manifest summary")
        raw = _bounded(self.predictions, _MAX_ITEMS, "predictions")
        predictions = []
        for item in raw:
            if type(item) is not DetectionCandidate:
                raise ValueError("predictions require DetectionCandidate values")
            copied = replace(item, box=_box(item.box))
            if copied.category not in WEAPON_TAXONOMY:
                raise ValueError("prediction category is outside the frozen taxonomy")
            predictions.append(copied)
        if type(self.generated_metadata) is not bool:
            raise ValueError("generated_metadata must be boolean")
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(self, "predictions", tuple(predictions))


@dataclass(frozen=True)
class WeaponDonorIdentity:
    donor_id: str
    code_repository: str
    code_revision: str
    model_sha256: str
    config_sha256: str
    runtime_name: str
    runtime_version: str
    runtime_sha256: str

    def __post_init__(self) -> None:
        for name in ("donor_id", "code_repository", "runtime_name", "runtime_version"):
            _text(getattr(self, name), name, 128)
        if type(self.code_revision) is not str or _REVISION.fullmatch(self.code_revision) is None:
            raise ValueError("code_revision must be a lowercase Git commit SHA-1")
        for name in ("model_sha256", "config_sha256", "runtime_sha256"):
            _digest(getattr(self, name), name)


@dataclass(frozen=True)
class WeaponClassMetrics:
    category: str
    true_positives: int
    false_positives: int
    false_negatives: int

    def __post_init__(self) -> None:
        if type(self.category) is not str or self.category not in WEAPON_TAXONOMY:
            raise ValueError("metrics category is outside the frozen taxonomy")
        for name in ("true_positives", "false_positives", "false_negatives"):
            _integer(getattr(self, name), name, _MAX_CASES * _MAX_ITEMS)

    @property
    def precision(self) -> float | None:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else None


@dataclass(frozen=True)
class WeaponEvaluationResult:
    donor: WeaponDonorIdentity
    dataset_sha256: str
    protocol_sha256: str
    case_count: int
    negative_images: int
    negative_images_with_false_positives: int
    negative_image_false_positives: int
    positive_images_with_misses: int
    duplicate_predictions: int
    per_class: tuple[WeaponClassMetrics, ...]
    generated_metadata: bool
    generated_cases: int

    def __post_init__(self) -> None:
        if type(self.donor) is not WeaponDonorIdentity:
            raise ValueError("result requires a WeaponDonorIdentity")
        object.__setattr__(self, "donor", replace(self.donor))
        _digest(self.dataset_sha256, "dataset_sha256")
        if self.protocol_sha256 != PROTOCOL_SHA256:
            raise ValueError("result protocol differs from the frozen Weapons protocol")
        _integer(self.case_count, "case_count", _MAX_CASES, 1)
        for name in ("negative_images", "negative_images_with_false_positives", "positive_images_with_misses"):
            _integer(getattr(self, name), name, self.case_count)
        for name in ("negative_image_false_positives", "duplicate_predictions"):
            _integer(getattr(self, name), name, self.case_count * _MAX_ITEMS)
        metrics = _bounded(self.per_class, len(WEAPON_TAXONOMY), "per-class metrics")
        if any(type(item) is not WeaponClassMetrics for item in metrics):
            raise ValueError("result requires typed metrics")
        metrics = tuple(replace(item) for item in metrics)
        if tuple(item.category for item in metrics) != WEAPON_TAXONOMY:
            raise ValueError("result requires each frozen class exactly once in taxonomy order")
        object.__setattr__(self, "per_class", metrics)
        tp, fp, fn = (sum(getattr(item, name) for item in metrics)
                      for name in ("true_positives", "false_positives", "false_negatives"))
        positive_images = self.case_count - self.negative_images
        if (tp + fp > self.case_count * _MAX_ITEMS or tp + fn > self.case_count * _MAX_ITEMS
                or not positive_images <= tp + fn <= positive_images * _MAX_ITEMS
                or not self.positive_images_with_misses <= fn <= self.positive_images_with_misses * _MAX_ITEMS
                or tp < positive_images - self.positive_images_with_misses
                or tp + fp - self.negative_image_false_positives > positive_images * _MAX_ITEMS
                or self.duplicate_predictions > fp - self.negative_image_false_positives
                or self.duplicate_predictions > _MAX_ITEMS * min(tp, positive_images) - tp
                or self.negative_images_with_false_positives > self.negative_images
                or self.positive_images_with_misses > self.case_count - self.negative_images
                or self.positive_images_with_misses > fn or self.duplicate_predictions > fp
                or self.negative_image_false_positives > fp
                or self.negative_image_false_positives < self.negative_images_with_false_positives
                or self.negative_image_false_positives > self.negative_images_with_false_positives * _MAX_ITEMS):
            raise ValueError("result counts are inconsistent")
        if type(self.generated_metadata) is not bool:
            raise ValueError("generated_metadata must be boolean")
        _integer(self.generated_cases, "generated_cases", self.case_count)
        if self.generated_metadata != (self.generated_cases == self.case_count):
            raise ValueError("generated metadata count/flag mismatch")

    @property
    def true_positives(self) -> int:
        return sum(item.true_positives for item in self.per_class)

    @property
    def false_positives(self) -> int:
        return sum(item.false_positives for item in self.per_class)

    @property
    def false_negatives(self) -> int:
        return sum(item.false_negatives for item in self.per_class)

    @property
    def precision(self) -> float | None:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else None

    def to_dict(self) -> dict:
        result = asdict(replace(self))
        result["per_class"] = [{**asdict(item), "precision": item.precision, "recall": item.recall}
                               for item in self.per_class]
        result["aggregate"] = {name: getattr(self, name) for name in
                               ("true_positives", "false_positives", "false_negatives", "precision", "recall")}
        result.update({"accuracy_claim": False, "readiness_claim": False, "identity_claim": False,
                       "authorizes_action": False, "protocol": dict(_PROTOCOL),
                       "mixed_generated_and_other_metadata": 0 < self.generated_cases < self.case_count,
                       "annotation_association": "caller_supplied_not_authenticated"})
        return result


def _match(ground_truth: tuple[WeaponAnnotationBox, ...], predictions: tuple[DetectionCandidate, ...]):
    edges = []
    for predicted in predictions:
        possible = [(index, _iou(predicted.box, labeled.box)) for index, labeled in enumerate(ground_truth)
                    if predicted.category == labeled.category]
        edges.append([index for index, _ in sorted(possible, key=lambda pair: (-pair[1], pair[0]))
                      if _ >= 0.5])
    owner: dict[int, int] = {}

    def augment(predicted_index: int, visited: set[int]) -> bool:
        for labeled_index in edges[predicted_index]:
            if labeled_index in visited:
                continue
            visited.add(labeled_index)
            if labeled_index not in owner or augment(owner[labeled_index], visited):
                owner[labeled_index] = predicted_index
                return True
        return False

    for index in range(len(predictions)):
        augment(index, set())
    return set(owner.values()), set(owner), edges


def evaluate_weapons(cases: Iterable[WeaponEvaluationCase], donor: WeaponDonorIdentity) -> WeaponEvaluationResult:
    if type(donor) is not WeaponDonorIdentity:
        raise ValueError("donor must be a WeaponDonorIdentity")
    donor = replace(donor)
    raw = _bounded(cases, _MAX_CASES, "cases")
    if not raw or any(type(item) is not WeaponEvaluationCase for item in raw):
        raise ValueError("evaluation requires 1-256 typed cases")
    cases = tuple(sorted((replace(item) for item in raw), key=lambda item: item.asset.asset_id))
    for field in ("asset_id", "sha256", "annotation_id"):
        if len({getattr(item.asset, field) for item in cases}) != len(cases):
            raise ValueError("duplicate evaluation asset/media/annotation identity")
    dataset = [{"asset": asdict(item.asset), "annotations": asdict(item.annotations),
                "generated_metadata": item.generated_metadata} for item in cases]
    counts = {category: [0, 0, 0] for category in WEAPON_TAXONOMY}
    negative_images = negative_alert_images = negative_fp = missed_images = duplicates = 0
    for case in cases:
        predictions = tuple(sorted((item for item in case.predictions if item.confidence >= 0.5),
                                   key=lambda item: (item.category, -item.confidence,
                                                     item.box.x_min, item.box.y_min, item.box.x_max, item.box.y_max)))
        gt = case.annotations.boxes
        matched_pred, matched_gt, edges = _match(gt, predictions)
        for index, item in enumerate(predictions):
            counts[item.category][0 if index in matched_pred else 1] += 1
            if index not in matched_pred and edges[index]:
                duplicates += 1
        for index, item in enumerate(gt):
            if index not in matched_gt:
                counts[item.category][2] += 1
        if not gt:
            negative_images += 1
            negative_fp += len(predictions)
            negative_alert_images += bool(predictions)
        elif len(matched_gt) < len(gt):
            missed_images += 1
    return WeaponEvaluationResult(donor, _hash(dataset), PROTOCOL_SHA256, len(cases), negative_images,
                                  negative_alert_images, negative_fp, missed_images, duplicates,
                                  tuple(WeaponClassMetrics(category, *counts[category]) for category in WEAPON_TAXONOMY),
                                  all(item.generated_metadata for item in cases),
                                  sum(item.generated_metadata for item in cases))


def compare_weapons(baseline: WeaponEvaluationResult, candidate: WeaponEvaluationResult) -> dict:
    if type(baseline) is not WeaponEvaluationResult or type(candidate) is not WeaponEvaluationResult:
        raise ValueError("comparison requires typed evaluation results")
    baseline, candidate = replace(baseline), replace(candidate)
    if ((baseline.dataset_sha256, baseline.protocol_sha256, baseline.case_count, baseline.negative_images,
         baseline.generated_metadata, baseline.generated_cases) != (candidate.dataset_sha256, candidate.protocol_sha256,
                                         candidate.case_count, candidate.negative_images, candidate.generated_metadata,
                                         candidate.generated_cases)
            or any(a.true_positives + a.false_negatives != b.true_positives + b.false_negatives
                   for a, b in zip(baseline.per_class, candidate.per_class))):
        raise ValueError("comparison requires identical held-out data and model-output evaluation protocol")
    delta = []
    for a, b in zip(baseline.per_class, candidate.per_class):
        row = {"category": a.category}
        for name in ("true_positives", "false_positives", "false_negatives", "precision", "recall"):
            av, bv = getattr(a, name), getattr(b, name)
            row[name] = None if av is None or bv is None else bv - av
        delta.append(row)
    return {"schema": "analytics.weapons-comparison.v1", "baseline": baseline.to_dict(),
            "candidate": candidate.to_dict(), "delta_candidate_minus_baseline": {
                "per_class": delta,
                "aggregate": {name: (None if getattr(baseline, name) is None or getattr(candidate, name) is None
                                      else getattr(candidate, name) - getattr(baseline, name))
                              for name in ("true_positives", "false_positives", "false_negatives", "precision", "recall")},
                **{name: getattr(candidate, name) - getattr(baseline, name) for name in
                   ("negative_image_false_positives", "negative_images_with_false_positives",
                    "positive_images_with_misses", "duplicate_predictions")}},
            "accuracy_claim": False, "readiness_claim": False, "authorizes_action": False,
            "automatic_promotion": False, "generated_metadata": baseline.generated_metadata}


def _synthetic_demo() -> dict:
    def make_case(name: str, positive: bool, predictions: tuple) -> WeaponEvaluationCase:
        media_hash = hashlib.sha256(("generated-metadata:" + name).encode()).hexdigest()
        boxes = (WeaponAnnotationBox("fixture-object", "rifle", NormalizedBox(0.1, 0.2, 0.9, 0.6)),) if positive else ()
        annotation = WeaponAnnotationPackage("fixture-annotation-" + name, media_hash, 100, 100, boxes)
        asset = WeaponTrainingAsset(name, "https://example.invalid/generated/" + name,
                                    "https://example.invalid/generated/" + name, "GENERATED-METADATA",
                                    "Generated metadata only; no media or model acquired", "rifle" if positive else None,
                                    "test", 1, media_hash, 100, 100, annotation.annotation_id, annotation.sha256,
                                    len(boxes), ("rifle",) if positive else (), True, True, False, True)
        return WeaponEvaluationCase(asset, annotation, predictions, True)

    def donor(name: str) -> WeaponDonorIdentity:
        digest = hashlib.sha256(("generated:" + name).encode()).hexdigest()
        return WeaponDonorIdentity(name, "generated-metadata-only", "0" * 40, digest, digest,
                                   "no-inference", "synthetic-fixture", digest)

    baseline_cases = (make_case("rifle", True, ()), make_case("negative", False, ()))
    candidate_cases = (make_case("rifle", True, (DetectionCandidate("rifle", 0.9, NormalizedBox(0.1, 0.2, 0.9, 0.6)),)),
                       make_case("negative", False, (DetectionCandidate("handgun", 0.8, NormalizedBox(0.1, 0.1, 0.3, 0.3)),)))
    return compare_weapons(evaluate_weapons(baseline_cases, donor("baseline-fixture")),
                           evaluate_weapons(candidate_cases, donor("candidate-fixture")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-demo", action="store_true", help="run generated metadata only, without inference")
    args = parser.parse_args()
    if not args.synthetic_demo:
        parser.error("specify --synthetic-demo; no media or donor execution is available")
    print(json.dumps(_synthetic_demo(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
