from __future__ import annotations

import unittest

from analytics_lab.figshare_acquisition import ZipMember
from analytics_lab.figshare_generalization import select_next_generalization_pair


class FigshareGeneralizationTests(unittest.TestCase):
    @staticmethod
    def member(name: str, size: int, offset: int) -> ZipMember:
        return ZipMember(name, size - 10, size, 8, offset & 0xFFFFFFFF, offset)

    def test_selects_smallest_hard_negative_and_distinct_fall_pair(self):
        members = (
            # Excluded because Location 3 was already used by the first pair.
            self.member(
                "VideoDataset/ADL/SBJ_02_LOC3/ACT19_R_1/negative-loc3.mp4",
                50,
                10,
            ),
            # Eligible ACT19 laying negative.
            self.member(
                "VideoDataset/ADL/SBJ_02_LOC1/ACT19_R_1/negative.mp4",
                100,
                20,
            ),
            # Smaller ACT4 member but same subject as the selected negative.
            self.member(
                "VideoDataset/Fall/SBJ_02_LOC2/ACT4_R_1/same-subject.mp4",
                40,
                30,
            ),
            # Smaller ACT4 member but same location as the selected negative.
            self.member(
                "VideoDataset/Fall/SBJ_03_LOC1/ACT4_R_1/same-location.mp4",
                45,
                40,
            ),
            # Smallest valid subject/location-disjoint positive.
            self.member(
                "VideoDataset/Fall/SBJ_03_LOC2/ACT4_R_1/positive.mp4",
                60,
                50,
            ),
            # Eligible but larger alternative negative/positive pair.
            self.member(
                "VideoDataset/ADL/SBJ_04_LOC2/ACT19_R_2/negative-large.mp4",
                180,
                60,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_05_LOC1/ACT4_R_2/positive-large.mp4",
                170,
                70,
            ),
            # Wrong activities never enter this evidence selection.
            self.member(
                "VideoDataset/ADL/SBJ_06_LOC1/ACT25_R_1/descend.mp4",
                10,
                80,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_07_LOC2/ACT10_R_1/chair-fall.mp4",
                10,
                90,
            ),
            # macOS sidecar is excluded by the exact dataset-path parser.
            self.member(
                "__MACOSX/VideoDataset/ADL/SBJ_08_LOC1/ACT19_R_1/._sidecar.mp4",
                5,
                100,
            ),
        )
        selected = select_next_generalization_pair(members)
        self.assertFalse(selected["commercial_accuracy_claim"])
        self.assertIn("no member payload", selected["selection_scope"])
        negative, positive = selected["members"]
        self.assertEqual(negative["activity_code"], "ACT19")
        self.assertEqual(negative["activity_name"], "Laying")
        self.assertEqual(negative["subject_id"], "02")
        self.assertEqual(negative["location_id"], "1")
        self.assertEqual(positive["activity_code"], "ACT4")
        self.assertEqual(positive["activity_name"], "Fall on the back")
        self.assertEqual(positive["subject_id"], "03")
        self.assertEqual(positive["location_id"], "2")
        self.assertNotEqual(negative["subject_id"], positive["subject_id"])
        self.assertNotEqual(negative["location_id"], positive["location_id"])

    def test_fails_closed_without_disjoint_pair(self):
        members = (
            self.member(
                "VideoDataset/ADL/SBJ_02_LOC1/ACT19_R_1/negative.mp4",
                100,
                20,
            ),
            self.member(
                "VideoDataset/Fall/SBJ_02_LOC2/ACT4_R_1/same-subject.mp4",
                60,
                30,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "subject/location-disjoint"):
            select_next_generalization_pair(members)

    def test_rejects_non_tuple_input(self):
        with self.assertRaises(TypeError):
            select_next_generalization_pair([])


if __name__ == "__main__":
    unittest.main()
