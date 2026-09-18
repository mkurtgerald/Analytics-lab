import unittest

from analytics_lab.perception import BBox
from analytics_lab.pose_crop import (
    DEFAULT_PADDING_FRACTION,
    OPENPOSE_MAX_ASPECT_RATIO,
    PoseCropPlan,
    plan_person_crop,
)


class PoseCropPlanTests(unittest.TestCase):
    def test_plan_expands_selection_with_bounded_context(self):
        plan = plan_person_crop(
            BBox(20, 20, 40, 60),
            frame_width=100,
            frame_height=100,
        )
        self.assertEqual(DEFAULT_PADDING_FRACTION, 0.20)
        self.assertEqual((plan.left, plan.top, plan.right, plan.bottom), (16, 12, 44, 68))
        self.assertEqual((plan.pad_top, plan.pad_bottom), (0, 0))
        self.assertEqual(plan.selection_bbox, BBox(4, 8, 24, 48))
        self.assertLessEqual(plan.output_width / plan.output_height, OPENPOSE_MAX_ASPECT_RATIO)

    def test_plan_uses_frame_context_before_black_padding_for_wide_person(self):
        plan = plan_person_crop(
            BBox(10, 0, 90, 20),
            frame_width=100,
            frame_height=40,
        )
        self.assertEqual((plan.left, plan.right), (0, 100))
        self.assertEqual((plan.top, plan.bottom), (0, 40))
        self.assertGreater(plan.pad_top + plan.pad_bottom, 0)
        self.assertLessEqual(plan.output_width / plan.output_height, OPENPOSE_MAX_ASPECT_RATIO)
        self.assertGreaterEqual(plan.selection_bbox.y1, plan.pad_top)
        self.assertLessEqual(plan.selection_bbox.y2, plan.output_height - plan.pad_bottom)

    def test_plan_rejects_selection_outside_frame(self):
        with self.assertRaises(ValueError):
            plan_person_crop(
                BBox(110, 10, 120, 20),
                frame_width=100,
                frame_height=100,
            )

    def test_plan_dataclass_rejects_empty_source_bounds(self):
        with self.assertRaises(ValueError):
            PoseCropPlan(
                left=5,
                top=5,
                right=5,
                bottom=10,
                pad_top=0,
                pad_bottom=0,
                selection_bbox=BBox(0, 0, 1, 1),
            )


if __name__ == "__main__":
    unittest.main()
