import ast
from pathlib import Path
import unittest


class AssociativeEmbeddingReferenceTests(unittest.TestCase):
    def _method(self, name):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "analytics_lab"
            / "associative_embedding_reference.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def test_match_by_tag_uses_group_axes_not_embedding_axis(self):
        method = self._method("_match_by_tag")
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

    def test_call_contract_is_spatially_dynamic_for_reviewed_176_outputs(self):
        method = self._method("__call__")
        rendered = ast.unparse(method)
        integer_constants = {
            node.value
            for node in ast.walk(method)
            if isinstance(node, ast.Constant) and type(node.value) is int
        }

        self.assertNotIn(144, integer_constants, "prior 0005 output size must not remain hard-coded")
        self.assertIn("heatmap_shape[0:2] != (1, 17)", rendered)
        self.assertIn("heatmap_shape[2] != heatmap_shape[3]", rendered)
        self.assertIn("tuple(tags.shape) != (1, 17, height, width, 1)", rendered)
        self.assertIn("tuple(nms_heatmaps.shape) != heatmap_shape", rendered)

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
