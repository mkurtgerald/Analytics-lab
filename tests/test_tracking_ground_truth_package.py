import hashlib
import json
import unittest

from analytics_lab.tracking_annotations import canonical_tracking_annotations_sha256
from analytics_lab.tracking_benchmark_plan import (
    TrackingBenchmarkPlan,
    wikimedia_pedestrian_smoke_plan,
)
from analytics_lab.tracking_evidence import canonical_frame_manifest_sha256
from analytics_lab.tracking_ground_truth_package import (
    canonical_rgb24_frame_sha256,
    parse_tracking_ground_truth_package,
    tracking_ground_truth_template_bytes,
)


def complete_payload(plan, *, objects=None):
    items = objects or {}
    return json.dumps({
        "schema_version": 1,
        "annotation_status": "complete_exhaustive",
        "benchmark_plan_sha256": plan.canonical_sha256(),
        "source_sha256": plan.source_sha256,
        "coordinate_space": "normalized_xyxy",
        "frames": [
            {
                "frame_index": index,
                "frame_sha256": hashlib.sha256(f"frame-{index}".encode()).hexdigest(),
                "objects": items.get(index, []),
            }
            for index in plan.frame_indices
        ],
    })


class TrackingGroundTruthPackageTests(unittest.TestCase):
    def test_complete_package_binds_annotations_and_frames(self):
        plan = wikimedia_pedestrian_smoke_plan()
        payload = complete_payload(plan, objects={
            plan.start_frame: [
                {"object_id": "p2", "category": "person", "box": [.3, .1, .4, .7]},
                {"object_id": "p1", "category": "person", "box": [.1, .1, .2, .7]},
            ]
        })
        result = parse_tracking_ground_truth_package(plan, payload)
        self.assertEqual(result.object_observations, 2)
        self.assertEqual(
            result.annotation_sha256,
            canonical_tracking_annotations_sha256(plan, result.frames),
        )
        self.assertEqual(
            result.frame_manifest_sha256,
            canonical_frame_manifest_sha256(
                result.frame_digests, max_frames=plan.frame_count
            ),
        )

    def test_template_is_prebound_but_intentionally_incomplete(self):
        plan = wikimedia_pedestrian_smoke_plan()
        template = tracking_ground_truth_template_bytes(plan)
        parsed = json.loads(template)
        self.assertEqual(parsed["benchmark_plan_sha256"], plan.canonical_sha256())
        self.assertEqual(
            [item["frame_index"] for item in parsed["frames"]],
            list(plan.frame_indices),
        )
        with self.assertRaises(ValueError):
            parse_tracking_ground_truth_package(plan, template)

    def test_wrong_plan_source_status_and_unknown_fields_fail_closed(self):
        plan = wikimedia_pedestrian_smoke_plan()
        for key, value in (
            ("annotation_status", "incomplete"),
            ("benchmark_plan_sha256", "0" * 64),
            ("source_sha256", "0" * 64),
            ("coordinate_space", "pixels_xyxy"),
        ):
            raw = json.loads(complete_payload(plan))
            raw[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_tracking_ground_truth_package(plan, json.dumps(raw))
        raw = json.loads(complete_payload(plan))
        raw["extra"] = True
        with self.assertRaises(ValueError):
            parse_tracking_ground_truth_package(plan, json.dumps(raw))

    def test_missing_reordered_or_unhashed_frame_fails_closed(self):
        plan = wikimedia_pedestrian_smoke_plan()
        raw = json.loads(complete_payload(plan))
        raw["frames"] = raw["frames"][:-1]
        with self.assertRaises(ValueError):
            parse_tracking_ground_truth_package(plan, json.dumps(raw))
        raw = json.loads(complete_payload(plan))
        raw["frames"][0], raw["frames"][1] = raw["frames"][1], raw["frames"][0]
        with self.assertRaises(ValueError):
            parse_tracking_ground_truth_package(plan, json.dumps(raw))
        raw = json.loads(complete_payload(plan))
        raw["frames"][0]["frame_sha256"] = ""
        with self.assertRaises(ValueError):
            parse_tracking_ground_truth_package(plan, json.dumps(raw))

    def test_object_schema_box_class_duplicate_and_bounds_fail_closed(self):
        plan = wikimedia_pedestrian_smoke_plan()
        frame = plan.start_frame
        invalid_objects = [
            [{"object_id": "p1", "category": "vehicle", "box": [.1, .1, .2, .4]}],
            [{"object_id": "p1", "category": "person", "box": [.2, .1, .1, .4]}],
            [{"object_id": "p1", "category": "person", "box": [.1, .1, .2]}],
            [{
                "object_id": "p1",
                "category": "person",
                "box": [.1, .1, .2, .4],
                "extra": 1,
            }],
            [
                {"object_id": "p1", "category": "person", "box": [.1, .1, .2, .4]},
                {"object_id": "p1", "category": "person", "box": [.3, .1, .4, .4]},
            ],
        ]
        for objects in invalid_objects:
            with self.subTest(objects=objects), self.assertRaises(ValueError):
                parse_tracking_ground_truth_package(
                    plan, complete_payload(plan, objects={frame: objects})
                )
        with self.assertRaises(RuntimeError):
            parse_tracking_ground_truth_package(
                plan,
                complete_payload(plan, objects={frame: [
                    {"object_id": "p1", "category": "person", "box": [.1, .1, .2, .4]},
                    {"object_id": "p2", "category": "person", "box": [.3, .1, .4, .4]},
                ]}),
                max_objects_per_frame=1,
            )

    def test_duplicate_json_keys_and_payload_bound_fail_closed(self):
        plan = wikimedia_pedestrian_smoke_plan()
        raw = complete_payload(plan)
        duplicate = raw[:-1] + ',"schema_version":1}'
        with self.assertRaises(ValueError):
            parse_tracking_ground_truth_package(plan, duplicate)
        with self.assertRaises(RuntimeError):
            parse_tracking_ground_truth_package(plan, raw, max_payload_bytes=10)

    def test_rgb24_digest_is_bound_to_frame_and_dimensions(self):
        plan = TrackingBenchmarkPlan(
            source_id="synthetic-contract-only",
            source_sha256="1" * 64,
            start_frame=7,
            frame_count=2,
            fps_numerator=25,
            fps_denominator=1,
            image_width=2,
            image_height=1,
            object_class="person",
            purpose="contract regression only",
            selection_basis="synthetic bytes exercise digest contract only",
        )
        pixels = bytes([1, 2, 3, 4, 5, 6])
        first = canonical_rgb24_frame_sha256(plan, 7, pixels)
        second = canonical_rgb24_frame_sha256(plan, 8, pixels)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, second)
        with self.assertRaises(ValueError):
            canonical_rgb24_frame_sha256(plan, 9, pixels)
        with self.assertRaises(ValueError):
            canonical_rgb24_frame_sha256(plan, 7, pixels[:-1])


if __name__ == "__main__":
    unittest.main()