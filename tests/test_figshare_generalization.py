from __future__ import annotations

import unittest

from analytics_lab.figshare_acquisition import ZipMember
from analytics_lab.figshare_generalization import select_next_generalization_pair


class FigshareGeneralizationTests(unittest.TestCase):
    @staticmethod
    def member(name: str, size: int, offset: int) -> ZipMember:
        return ZipMember(name, size - 10, size, 8, offset & 0xFFFFFFFF, offset)

    def test_selects_forward_bending_boundary_with_untouched_subject(self):
        members = (
            # Excluded because Subject 07 was used by the fourth admitted pair.
            self.member(
                "VideoDataset/ADL/SBJ_07_LOC4/ACT18_R_1/excluded-subject.mp4",
                20,
                10,
            ),
            # Eligible ACT18 picking-up negative from a new location.
            self.member(
                "VideoDataset/ADL/SBJ_04_LOC4/ACT18_R_1/negative.mp4",
                100,
                20,
            ),
            # Smaller ACT3 member but the same subject as the selected negative.
            self.member(
                "VideoDataset/Fall/SBJ_04_LOC1/ACT3_R_1/same-subject.mp4",
                30,
                30,
            ),
            # Same-location candidate is invalid and deliberately larger so it
            # cannot become a cheaper valid cross-pair with another fixture row.
            self.member(
                "VideoDataset/Fall/SBJ_05_LOC4/ACT3_R_1/same-location.mp4",
                500,
                40,
            ),
            # Smallest valid subject/location-disjoint positive paired with loc4.
            self.member(
                "VideoDataset/Fall/SBJ_05_LOC1/ACT3_R_1/positive.mp4",
                60,
                50,
            ),
            # Eligible but larger alternative positive.
            self.member(
                "VideoDataset/Fall/SBJ_08_LOC2/ACT3_R_2/positive-large.mp4",
                170,
                60,
            ),
            # Wrong activities never enter this evidence selection.
            self.member(
                "VideoDataset/ADL/SBJ_08_LOC4/ACT16_R_1/sitting.mp4",
                10,
                70,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_11_LOC1/ACT11_R_1/sit-fall.mp4",
                10,
                80,
            ),
            # macOS sidecar is excluded by the exact dataset-path parser.
            self.member(
                "__MACOSX/VideoDataset/ADL/SBJ_12_LOC4/ACT18_R_1/._sidecar.mp4",
                5,
                90,
            ),
        )
        selected = select_next_generalization_pair(members)
        self.assertFalse(selected["commercial_accuracy_claim"])
        self.assertIn("no member payload", selected["selection_scope"])
        policy = selected["selection_policy"]
        self.assertFalse(policy["require_at_least_one_novel_location"])
        self.assertEqual(policy["excluded_subject_ids"], ["01", "02", "03", "06", "07", "09", "10", "29"])
        self.assertEqual(policy["previously_used_location_ids"], ["1", "2", "3", "5"])
        negative, positive = selected["members"]
        self.assertEqual(negative["activity_code"], "ACT18")
        self.assertEqual(negative["activity_name"], "Picking up")
        self.assertEqual(negative["subject_id"], "04")
        self.assertEqual(negative["location_id"], "4")
        self.assertEqual(positive["activity_code"], "ACT3")
        self.assertEqual(positive["activity_name"], "Fall on the front")
        self.assertEqual(positive["subject_id"], "05")
        self.assertEqual(positive["location_id"], "1")
        self.assertNotEqual(negative["subject_id"], positive["subject_id"])
        self.assertNotEqual(negative["location_id"], positive["location_id"])

    def test_prefers_more_novel_location_coverage_before_byte_size(self):
        members = (
            self.member(
                "VideoDataset/ADL/SBJ_04_LOC4/ACT18_R_1/negative-new.mp4",
                200,
                20,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_05_LOC6/ACT3_R_1/positive-new.mp4",
                200,
                30,
            ),
            self.member(
                "VideoDataset/ADL/SBJ_08_LOC4/ACT18_R_1/negative-one-new.mp4",
                30,
                40,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_11_LOC1/ACT3_R_1/positive-old.mp4",
                30,
                50,
            ),
        )
        selected = select_next_generalization_pair(members)
        negative, positive = selected["members"]
        self.assertEqual({negative["location_id"], positive["location_id"]}, {"4", "6"})

    def test_accepts_subject_location_disjoint_pair_when_no_novel_location_exists(self):
        members = (
            self.member(
                "VideoDataset/ADL/SBJ_04_LOC1/ACT18_R_1/negative.mp4",
                100,
                20,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_05_LOC2/ACT3_R_1/positive.mp4",
                60,
                30,
            ),
        )
        selected = select_next_generalization_pair(members)
        negative, positive = selected["members"]
        self.assertEqual((negative["location_id"], positive["location_id"]), ("1", "2"))
        self.assertFalse(selected["selection_policy"]["require_at_least_one_novel_location"])

    def test_fails_closed_without_subject_location_disjoint_pair(self):
        members = (
            self.member(
                "VideoDataset/ADL/SBJ_04_LOC1/ACT18_R_1/negative.mp4",
                100,
                20,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_04_LOC2/ACT3_R_1/same-subject.mp4",
                60,
                30,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_05_LOC1/ACT3_R_1/same-location.mp4",
                70,
                40,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "subject/location-disjoint"):
            select_next_generalization_pair(members)

    def test_rejects_non_tuple_input(self):
        with self.assertRaises(TypeError):
            select_next_generalization_pair([])


if __name__ == "__main__":
    unittest.main()
