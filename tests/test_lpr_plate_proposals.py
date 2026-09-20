import unittest
from analytics_lab.lpr_plate_proposals import PlateProposalConfig, rank_plate_rectangles


class PlateProposalTests(unittest.TestCase):
    def test_ranks_plate_like_rectangle_and_rejects_wrong_geometry(self):
        result = rank_plate_rectangles(
            (
                (100, 200, 400, 100, 32000.0),
                (20, 30, 50, 300, 12000.0),
                (0, 0, 900, 700, 500000.0),
            ),
            image_width=1000,
            image_height=800,
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].x1, 0.1)
        self.assertAlmostEqual(result[0].x2, 0.5)
        self.assertGreater(result[0].score, 0.7)

    def test_non_max_suppression_keeps_best_overlap(self):
        result = rank_plate_rectangles(
            (
                (100, 200, 400, 100, 32000.0),
                (105, 202, 390, 98, 25000.0),
                (600, 300, 240, 70, 12000.0),
            ),
            image_width=1000,
            image_height=800,
        )
        self.assertEqual(len(result), 2)
        self.assertLess(result[0].x1, result[1].x1)

    def test_invalid_config_fails_closed(self):
        with self.assertRaises(ValueError):
            PlateProposalConfig(min_aspect_ratio=5.0, target_aspect_ratio=4.2)

    def test_bounds_candidate_count(self):
        cfg = PlateProposalConfig(max_plates=2)
        result = rank_plate_rectangles(
            tuple((10 + i * 200, 100, 160, 40, 5000.0) for i in range(4)),
            image_width=1000,
            image_height=400,
            config=cfg,
        )
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
