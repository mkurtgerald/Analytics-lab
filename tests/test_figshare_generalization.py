from __future__ import annotations

import unittest

from analytics_lab.figshare_acquisition import ZipMember
from analytics_lab.figshare_generalization import select_next_generalization_pair


class FigshareGeneralizationTests(unittest.TestCase):
    @staticmethod
    def member(name: str, size: int, offset: int) -> ZipMember:
        return ZipMember(name, size - 10, size, 8, offset & 0xFFFFFFFF, offset)

    def test_selects_sitting_boundary_with_untouched_subject_and_novel_location(self):
        members = (
            # Excluded subjects were used by the first two admitted pairs.
            self.member(
                "VideoDataset/ADL/SBJ_06_LOC4/ACT16_R_1/excluded-subject.mp4",
                20,
                10,
            ),
            # A smaller pair across only previously exercised locations must
            # lose to a pair that expands environmental coverage.
            self.member(
                "VideoDataset/ADL/SBJ_02_LOC1/ACT16_R_1/old-location-negative.mp4",
                40,
                20,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_04_LOC2/ACT11_R_1/old-location-positive.mp4",
                40,
                30,
            ),
            # Eligible sitting negative from a new location.
            self.member(
                "VideoDataset/ADL/SBJ_02_LOC4/ACT16_R_1/negative.mp4",
                100,
                40,
            ),
            # Smaller ACT11 candidate but same subject as the selected negative.
            self.member(
                "VideoDataset/Fall/SBJ_02_LOC1/ACT11_R_1/same-subject.mp4",
                30,
                50,
            ),
            # Same-location candidate is also invalid.
            self.member(
                "VideoDataset/Fall/SBJ_05_LOC4/ACT11_R_1/same-location.mp4",
                35,
                60,
            ),
            # Smallest valid subject/location-disjoint positive paired with loc4.
            self.member(
                "VideoDataset/Fall/SBJ_04_LOC1/ACT11_R_1/positive.mp4",
                60,
                70,
            ),
            # Larger valid alternative.
            self.member(
                "VideoDataset/Fall/SBJ_05_LOC2/ACT11_R_2/positive-large.mp4",
                170,
                80,
            ),
            # Wrong activities never enter this evidence selection.
            self.member(
                "VideoDataset/ADL/SBJ_07_LOC4/ACT19_R_1/laying.mp4",
                10,
                90,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_08_LOC1/ACT4_R_1/back-fall.mp4",
                10,
                100,
            ),
            # macOS sidecars are excluded by the exact dataset-path parser.
            self.member(
                "__MACOSX/VideoDataset/ADL/SBJ_09_LOC4/ACT16_R_1/._sidecar.mp4",
                5,
                110,
            ),
        )
        selected = select_next_generalization_pair(members)
        self.assertFalse(selected["commercial_accuracy_claim"])
        self.assertIn("no member payload", selected["selection_scope"])
        policy = selected["selection_policy"]
        self.assertTrue(policy["require_at_least_one_novel_location"])
        self.assertEqual(policy["excluded_subject_ids"], ["01", "03", "06", "10"])
        negative, positive = selected["members"]
        self.assertEqual(negative["activity_code"], "ACT16")
        self.assertEqual(negative["activity_name"], "Sitting")
        self.assertEqual(negative["subject_id"], "02")
        self.assertEqual(negative["location_id"], "4")
        self.assertEqual(positive["activity_code"], "ACT11")
        self.assertEqual(positive["activity_name"], "Try to sit on chair, fall")
        self.assertEqual(positive["subject_id"], "04")
        self.assertEqual(positive["location_id"], "1")
        self.assertNotEqual(negative["subject_id"], positive["subject_id"])
        self.assertNotEqual(negative["location_id"], positive["location_id"])

    def test_prefers_more_novel_location_coverage_before_byte_size(self):
        members = (
            self.member(
                "VideoDataset/ADL/SBJ_02_LOC4/ACT16_R_1/negative-new.mp4",
                200,
                20,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_04_LOC5/ACT11_R_1/positive-new.mp4",
                200,
                30,
            ),
            self.member(
                "VideoDataset/ADL/SBJ_05_LOC4/ACT16_R_1/negative-one-new.mp4",
                30,
                40,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_07_LOC1/ACT11_R_1/positive-old.mp4",
                30,
                50,
            ),
        )
        selected = select_next_generalization_pair(members)
        negative, positive = selected["members"]
        self.assertEqual({negative["location_id"], positive["location_id"]}, {"4", "5"})

    def test_fails_closed_without_novel_location_pair(self):
        members = (
            self.member(
                "VideoDataset/ADL/SBJ_02_LOC1/ACT16_R_1/negative.mp4",
                100,
                20,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_04_LOC2/ACT11_R_1/positive.mp4",
                60,
                30,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "novel location"):
            select_next_generalization_pair(members)

    def test_rejects_non_tuple_input(self):
        with self.assertRaises(TypeError):
            select_next_generalization_pair([])


if __name__ == "__main__":
    unittest.main()
