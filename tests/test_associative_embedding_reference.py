import unittest

import numpy as np

from analytics_lab.associative_embedding_reference import AssociativeEmbeddingDecoder


class AssociativeEmbeddingReferenceTests(unittest.TestCase):
    def test_match_by_tag_uses_group_axes_not_embedding_axis(self):
        decoder = AssociativeEmbeddingDecoder()
        decoder._max_match = lambda scores: np.array([[0, 0]], dtype=np.int64)

        tag_k = np.zeros((17, 1, 1), dtype=np.float32)
        loc_k = np.zeros((17, 1, 2), dtype=np.int64)
        val_k = np.zeros((17, 1), dtype=np.float32)
        tag_k[0, 0, 0] = 0.25
        tag_k[1, 0, 0] = 0.25
        loc_k[0, 0] = (10, 10)
        loc_k[1, 0] = (11, 10)
        val_k[0, 0] = 0.9
        val_k[1, 0] = 0.8

        poses, pose_tags = decoder._match_by_tag((tag_k, loc_k, val_k))

        self.assertEqual(poses.shape, (1, 17, 4))
        self.assertEqual(pose_tags.shape, (1, 1))
        self.assertAlmostEqual(float(poses[0, 0, 2]), 0.9, places=6)
        self.assertAlmostEqual(float(poses[0, 1, 2]), 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
