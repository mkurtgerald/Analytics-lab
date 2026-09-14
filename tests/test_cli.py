from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def replay(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_bytes(content)
            return subprocess.run(
                [sys.executable, "-m", "analytics_lab", "--input", str(path),
                 "--source-id", "synthetic", "--session-id", "test"],
                cwd=ROOT, capture_output=True, text=True, timeout=15, check=False,
            )

    def test_example_emits_one_candidate(self):
        result = self.replay((ROOT / "examples/person_down.jsonl").read_bytes())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)["status"], "candidate")

    def test_empty_input_emits_nothing(self):
        result = self.replay(b"\n\n")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_invalid_json_rejected_without_echo(self):
        result = self.replay(b"super-secret-not-json")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("super-secret", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_non_object_json_rejected(self):
        self.assertEqual(self.replay(b"[]\n").returncode, 2)

    def test_missing_fields_rejected(self):
        self.assertEqual(self.replay(b"{}\n").returncode, 2)

    def test_oversized_line_rejected(self):
        result = self.replay(b"x" * 16385)
        self.assertEqual(result.returncode, 2)
        self.assertLess(len(result.stderr), 200)

    def test_nan_rejected(self):
        result = self.replay(b'{"timestamp_ms":0,"track_id":"x","posture":"down","confidence":NaN}\n')
        self.assertEqual(result.returncode, 2)

    def test_out_of_order_is_nonzero(self):
        result = self.replay(b'{"timestamp_ms":10,"track_id":"x","posture":"down","confidence":0.9}\n'
                             b'{"timestamp_ms":0,"track_id":"x","posture":"down","confidence":0.9}\n')
        self.assertEqual(result.returncode, 2)
        self.assertIn("line 2", result.stderr)


if __name__ == "__main__":
    unittest.main()
