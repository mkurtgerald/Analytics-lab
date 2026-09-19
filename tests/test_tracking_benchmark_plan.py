import unittest

from analytics_lab.tracking_benchmark_plan import (
    TrackingBenchmarkPlan,
    WIKIMEDIA_PEDESTRIAN_SOURCE_SHA256,
    wikimedia_pedestrian_smoke_plan,
)


class TrackingBenchmarkPlanTests(unittest.TestCase):
    def test_wikimedia_plan_is_exact_and_small(self):
        plan = wikimedia_pedestrian_smoke_plan()
        self.assertEqual(plan.source_sha256, WIKIMEDIA_PEDESTRIAN_SOURCE_SHA256)
        self.assertEqual(plan.frame_indices, tuple(range(150, 175)))
        self.assertEqual(float(plan.start_seconds), 6.0)
        self.assertEqual(float(plan.last_frame_seconds), 6.96)
        self.assertEqual(plan.object_class, "person")
        self.assertEqual(
            plan.canonical_sha256(),
            "eb7995a389a22f3528b9bfca97d64f9d8e3c515ebf1c764cf25b68fbdcc3479c",
        )

    def test_plan_hash_is_deterministic_and_binds_window(self):
        plan = wikimedia_pedestrian_smoke_plan()
        same = wikimedia_pedestrian_smoke_plan()
        self.assertEqual(plan.canonical_sha256(), same.canonical_sha256())
        changed = TrackingBenchmarkPlan(**{**plan.__dict__, "start_frame": 151})
        self.assertNotEqual(plan.canonical_sha256(), changed.canonical_sha256())

    def test_plan_rejects_unbound_or_unbounded_inputs(self):
        plan = wikimedia_pedestrian_smoke_plan()
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            TrackingBenchmarkPlan(**{**plan.__dict__, "source_sha256": "bad"})
        with self.assertRaisesRegex(ValueError, "frame_count"):
            TrackingBenchmarkPlan(**{**plan.__dict__, "frame_count": 501})
        with self.assertRaisesRegex(ValueError, "start_frame"):
            TrackingBenchmarkPlan(**{**plan.__dict__, "start_frame": -1})


if __name__ == "__main__":
    unittest.main()
