import unittest

from analytics_lab.uvify_tracking import (
    UVIFY_LICENSE,
    UVIFY_REPOSITORY,
    UVIFY_REVISION,
    parse_uvify_ground_truth,
)


class UvifyTrackingTests(unittest.TestCase):
    def test_parses_valid_rows_and_uses_tracking_id_not_person_id(self):
        labels = "\n".join((
            "450000,7,11,100,50,200,300,1,0,1,1,0.8",
            "450000,8,12,500,100,100,200,1,1,0,1,0.5",
            # Same person_id later receives a different dataset tracking_id.
            "450001,7,19,120,60,200,300,1,2,1,1,0.7",
        ))
        frames = parse_uvify_ground_truth(labels, 1000, 500)
        self.assertEqual([frame.frame_index for frame in frames], [450000, 450001])
        self.assertEqual(
            [item.object_id for item in frames[0].ground_truth],
            ["uvify-track:11", "uvify-track:12"],
        )
        self.assertEqual(frames[1].ground_truth[0].object_id, "uvify-track:19")
        first = frames[0].ground_truth[0].box
        self.assertEqual((first.x_min, first.y_min, first.x_max, first.y_max), (.1, .1, .3, .7))

    def test_invalid_source_rows_are_not_silently_repaired(self):
        with self.assertRaisesRegex(ValueError, "expected 12"):
            parse_uvify_ground_truth("1,2,3", 100, 100)
        with self.assertRaisesRegex(ValueError, "outside"):
            parse_uvify_ground_truth("1,2,3,90,10,20,20,1,0,1,1,0.5", 100, 100)
        with self.assertRaisesRegex(ValueError, "visibility"):
            parse_uvify_ground_truth("1,2,3,10,10,20,20,1,0,1,1,1.5", 100, 100)
        with self.assertRaisesRegex(ValueError, "pose_class"):
            parse_uvify_ground_truth("1,2,3,10,10,20,20,1,9,1,1,0.5", 100, 100)

    def test_invalid_annotations_are_excluded_but_empty_admission_fails(self):
        labels = "\n".join((
            "1,2,3,10,10,20,20,0,0,1,1,0.5",
            "2,2,4,20,20,20,20,1,0,1,1,0.5",
        ))
        frames = parse_uvify_ground_truth(labels, 100, 100)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].frame_index, 2)
        self.assertEqual(frames[0].ground_truth[0].object_id, "uvify-track:4")
        with self.assertRaisesRegex(ValueError, "no valid"):
            parse_uvify_ground_truth("1,2,3,10,10,20,20,0,0,1,1,0.5", 100, 100)

    def test_duplicate_tracking_id_in_frame_fails_closed(self):
        labels = "\n".join((
            "1,2,3,10,10,20,20,1,0,1,1,0.5",
            "1,8,3,40,10,20,20,1,0,1,1,0.5",
        ))
        with self.assertRaisesRegex(ValueError, "duplicate tracking_id"):
            parse_uvify_ground_truth(labels, 100, 100)

    def test_configured_bounds_are_enforced(self):
        labels = "\n".join((
            "1,2,3,10,10,20,20,1,0,1,1,0.5",
            "1,4,5,40,10,20,20,1,0,1,1,0.5",
        ))
        with self.assertRaisesRegex(RuntimeError, "row count"):
            parse_uvify_ground_truth(labels, 100, 100, max_rows=1)
        with self.assertRaisesRegex(RuntimeError, "object count"):
            parse_uvify_ground_truth(labels, 100, 100, max_objects_per_frame=1)

    def test_source_identity_is_pinned(self):
        self.assertEqual(UVIFY_REPOSITORY, "uvify-public/human_tracking_dataset")
        self.assertEqual(UVIFY_REVISION, "eb3af0cfe49de018a0c4736581daadd8eb860883")
        self.assertEqual(UVIFY_LICENSE, "CC-BY-4.0")


if __name__ == "__main__":
    unittest.main()
