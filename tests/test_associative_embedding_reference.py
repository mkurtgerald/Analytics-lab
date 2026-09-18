import ast
from pathlib import Path
import unittest

import numpy as np

from analytics_lab.associative_embedding_reference import AssociativeEmbeddingDecoder


class AssociativeEmbeddingReferenceTests(unittest.TestCase):
    def test_match_by_tag_uses_group_axes_not_embedding_axis(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "analytics_lab"
            / "associative_embedding_reference.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_match_by_tag"
        )

        assignments = {}
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                assignments[node.targets[0].id] = node.value

        self._assert_diff_shape_axis(assignments.get("num_added"), 0)
        self._assert_diff_shape_axis(assignments.get("num_grouped"), 1)

        for node in ast.walk(method):
            if not isinstance(node, ast.Assign):
                continue
            self.assertFalse(
                any(
                    isinstance(target, (ast.Tuple, ast.List))
                    and {item.id for item in target.elts if isinstance(item, ast.Name)}
                    >= {"num_added", "num_grouped"}
                    for target in node.targets
                )
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "diff"
                and node.value.attr == "shape",
                "group counts must not unpack the three-axis diff.shape tuple",
            )

    def test_call_accepts_reviewed_176_square_contract(self):
        decoder = AssociativeEmbeddingDecoder()
        decoder.max_num_people = 1
        heatmaps = np.zeros((1, 17, 176, 176), dtype=np.float32)
        tags = np.zeros((1, 17, 176, 176, 1), dtype=np.float32)

        poses, scores = decoder(heatmaps, tags)

        self.assertEqual(poses.shape, (0, 17, 4))
        self.assertEqual(scores.shape, (0,))

    def test_call_rejects_mismatched_spatial_contract(self):
        decoder = AssociativeEmbeddingDecoder()
        heatmaps = np.zeros((1, 17, 176, 176), dtype=np.float32)
        tags = np.zeros((1, 17, 144, 144, 1), dtype=np.float32)

        with self.assertRaisesRegex(RuntimeError, "embedding output contract"):
            decoder(heatmaps, tags)

    def _assert_diff_shape_axis(self, expression, axis):
        self.assertIsInstance(expression, ast.Subscript)
        self.assertIsInstance(expression.value, ast.Attribute)
        self.assertEqual(expression.value.attr, "shape")
        self.assertIsInstance(expression.value.value, ast.Name)
        self.assertEqual(expression.value.value.id, "diff")
        self.assertIsInstance(expression.slice, ast.Constant)
        self.assertEqual(expression.slice.value, axis)


if __name__ == "__main__":
    unittest.main()
