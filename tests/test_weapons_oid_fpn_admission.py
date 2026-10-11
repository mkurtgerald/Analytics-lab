import copy
import gzip
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analytics_lab import weapons_oid_fpn_admission as admission
from analytics_lab import weapons_oid_fpn_discovery as discovery
from tests.test_weapons_oid_fpn_discovery import archive, Clock, OBSERVED_PAX, pax_control, Response


GRAPH = b"generated-graph-fixture" * 4000
CONFIG = b"model { ssd { num_classes: 601 image_resizer { fixed_shape_resizer { height: 512 width: 512 } } } }"


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.raw = pax_control(OBSERVED_PAX) + archive(((discovery._ROOT, None),
                    (discovery._ROOT + "/frozen_inference_graph.pb", GRAPH),
                    (discovery._ROOT + "/pipeline.config", CONFIG),
                    (discovery._ROOT + "/unused-weights", b"generated-unused")))
        self.payload = gzip.compress(self.raw, mtime=0)
        self.size_patch = patch.object(discovery, "_EXPECTED_SIZE", len(self.payload))
        self.size_patch.start()
        self.addCleanup(self.size_patch.stop)
        self.calls, self.responses = 0, []
        self.expected = discovery.run(opener=self.opener, clock=Clock())
        self.calls, self.responses = 0, []

    def opener(self, request, *, timeout):
        self.calls += 1
        response = Response(self.payload)
        self.responses.append(response)
        return response

    def fixture_pin(self, expected=None):
        value = self.expected if expected is None else expected
        return patch.object(admission, "identity", side_effect=lambda: copy.deepcopy(value))

    def test_real_bundled_pin_preserves_complete_hosted_identity(self):
        expected = admission.identity()
        self.assertEqual(expected["archive_size"], 638212759)
        self.assertEqual(expected["archive_sha256"], "b0fe7ae2a91a29ced95e564141d2a8aadfd6da2d7da9565395c4841011d4ea75")
        self.assertEqual(expected["expanded_size"], 710701056)
        self.assertEqual(expected["member_count"], 20)
        self.assertEqual(len(expected["members"]), 20)
        self.assertEqual(expected["graph"]["size"], 235714499)
        self.assertEqual(expected["graph"]["sha256"], "0c9269fff044e0126bd43c3496c738d3445603ffb7fe07905a3d1149fc6446d3")
        config = next(member for member in expected["members"] if member["type"] == "file" and member["name"].endswith("/pipeline.config"))
        self.assertEqual(config["sha256"], "5898eb7d03d6bcecbc368372c910842e9f19f6f9ac5510f15b18b1da92ef9d13")
        self.assertEqual(expected["pipeline"]["image_resizer"], {"type": "fixed_shape_resizer", "height": 512, "width": 512})
        expected["members"][0]["fields"]["uid"] = "changed"
        self.assertEqual(admission.identity()["members"][0]["fields"]["uid"], "345018")

    def test_graph_yielded_after_one_complete_verified_stream_and_only_graph_written(self):
        with tempfile.TemporaryDirectory() as runner, patch.dict(os.environ, RUNNER_TEMP=runner), self.fixture_pin():
            work = Path(runner) / "new-parent" / "new-child"
            with admission.verified_graph(work, opener=self.opener, clock=Clock()) as graph:
                self.assertEqual(graph.read_bytes(), GRAPH)
                self.assertEqual([item for item in Path(runner).rglob("*") if item.is_file()], [graph])
                self.assertEqual(self.calls, 1)
                self.assertTrue(self.responses[0].closed)
                self.assertEqual(self.responses[0].payload.tell(), len(self.payload))
                self.assertEqual(hashlib.sha256(graph.read_bytes()).hexdigest(), self.expected["graph"]["sha256"])
            self.assertFalse(graph.exists())
            self.assertEqual(list(Path(runner).iterdir()), [])

    def test_existing_parent_and_unrelated_files_preserved_and_paths_unique(self):
        with tempfile.TemporaryDirectory() as runner, patch.dict(os.environ, RUNNER_TEMP=runner), self.fixture_pin():
            work = Path(runner) / "existing"
            work.mkdir()
            sentinel = work / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with admission.verified_graph(work, opener=self.opener, clock=Clock()) as first:
                with admission.verified_graph(work, opener=self.opener, clock=Clock()) as second:
                    self.assertNotEqual(first, second)
                    self.assertTrue(first.exists())
                    self.assertTrue(second.exists())
            self.assertEqual(list(work.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_consumer_failure_cleans_graph(self):
        with tempfile.TemporaryDirectory() as runner, patch.dict(os.environ, RUNNER_TEMP=runner), self.fixture_pin():
            with self.assertRaisesRegex(RuntimeError, "consumer failure"):
                with admission.verified_graph(Path(runner), opener=self.opener, clock=Clock()) as graph:
                    raise RuntimeError("consumer failure")
            self.assertFalse(graph.exists())
            self.assertEqual(list(Path(runner).iterdir()), [])

    def test_every_pinned_report_boundary_checked_before_yield(self):
        def changed(field, value):
            result = copy.deepcopy(self.expected)
            result[field] = value
            return result
        variants = [changed("archive_sha256", "0"*64), changed("archive_size", len(self.payload)+1),
                    changed("expanded_size", len(self.raw)+512), changed("archive_root", "other-root"),
                    changed("member_count", self.expected["member_count"]+1),
                    changed("members", list(reversed(self.expected["members"]))),
                    changed("oid_v4_label_map_blob", "0"*40)]
        for target, key, value in (("graph", "sha256", "0"*64), ("pipeline", "num_classes", 600)):
            expected = copy.deepcopy(self.expected)
            expected[target][key] = value
            variants.append(expected)
        for field, value in (("payload_utf8", "14 uid=345019\n27 mtime=1548178193.679631\n"),
                             ("sha256", "0"*64), ("following_member", "other-member"),
                             ("original_name", "other-control")):
            expected = copy.deepcopy(self.expected)
            expected["members"][0][field] = value
            variants.append(expected)
        expected = copy.deepcopy(self.expected)
        next(member for member in expected["members"] if member["name"].endswith("/pipeline.config"))["sha256"] = "0"*64
        variants.append(expected)
        for index, expected in enumerate(variants):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as runner, patch.dict(os.environ, RUNNER_TEMP=runner), self.fixture_pin(expected):
                yielded = False
                with self.assertRaises(RuntimeError):
                    with admission.verified_graph(Path(runner), opener=self.opener, clock=Clock()):
                        yielded = True
                self.assertFalse(yielded)
                self.assertEqual(list(Path(runner).iterdir()), [])

    def test_pinned_graph_header_mismatch_rejected_before_yield(self):
        for field, value in (("name", "other/graph"), ("original_name", "./different"), ("size", len(GRAPH)-1)):
            expected = copy.deepcopy(self.expected)
            expected["graph"][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as runner, patch.dict(os.environ, RUNNER_TEMP=runner), self.fixture_pin(expected):
                with self.assertRaisesRegex(RuntimeError, "streamed graph"):
                    with admission.verified_graph(Path(runner), opener=self.opener, clock=Clock()):
                        self.fail("unverified graph was yielded")
                self.assertEqual(list(Path(runner).iterdir()), [])

    def test_truncated_stream_cleanup(self):
        self.payload = self.payload[:-6]
        with tempfile.TemporaryDirectory() as runner, patch.dict(os.environ, RUNNER_TEMP=runner), self.fixture_pin(), patch.object(discovery, "_EXPECTED_SIZE", len(self.payload)):
            with self.assertRaises(RuntimeError):
                with admission.verified_graph(Path(runner), opener=self.opener, clock=Clock()):
                    self.fail("truncated archive yielded a graph")
            self.assertEqual(list(Path(runner).iterdir()), [])

    def test_missing_runner_outside_paths_symlink_and_non_directory_rejected(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            with admission.verified_graph(Path("/tmp"), opener=self.opener):
                self.fail("missing runner accepted")
        with tempfile.TemporaryDirectory() as runner, tempfile.TemporaryDirectory() as outside, patch.dict(os.environ, RUNNER_TEMP=runner):
            file_path = Path(runner) / "file"
            file_path.write_bytes(b"preserve")
            paths = (Path(outside), Path(runner) / ".." / "escape", Path("relative"), file_path)
            for parent in paths:
                with self.subTest(parent=parent), self.assertRaises(RuntimeError):
                    with admission.verified_graph(parent, opener=self.opener):
                        self.fail("unsafe parent accepted")
            link = Path(runner) / "link"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                return  # Windows runner without symlink privilege still exercises other path guards.
            with self.assertRaises(RuntimeError):
                with admission.verified_graph(link / "child", opener=self.opener):
                    self.fail("symlink parent accepted")
            with patch.dict(os.environ, RUNNER_TEMP=str(link)), self.assertRaises(RuntimeError):
                with admission.verified_graph(link, opener=self.opener):
                    self.fail("symlink RUNNER_TEMP accepted")
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_bundled_identity_corruption_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            pin = Path(root) / "pin.json"
            pin.write_bytes(b"{}")
            with patch.object(admission, "_IDENTITY_PATH", pin), self.assertRaises(RuntimeError):
                admission.identity()
            target = Path(root) / "link.json"
            try:
                target.symlink_to(admission._IDENTITY_PATH)
            except OSError:
                return
            with patch.object(admission, "_IDENTITY_PATH", target), self.assertRaises(RuntimeError):
                admission.identity()

    def test_post_stream_file_verification_deadline_prevents_yield(self):
        clock = Clock()
        original = discovery._run
        def late_stream(**kwargs):
            result = original(**kwargs)
            clock.now = discovery._TIME_LIMIT + 1
            return result
        with tempfile.TemporaryDirectory() as runner, patch.dict(os.environ, RUNNER_TEMP=runner), self.fixture_pin(), patch.object(discovery, "_run", side_effect=late_stream):
            with self.assertRaises(TimeoutError):
                with admission.verified_graph(Path(runner), opener=self.opener, clock=clock):
                    self.fail("late graph yielded")
            self.assertEqual(list(Path(runner).iterdir()), [])

    def test_corrupted_materialized_graph_rejected_after_archive_identity_matches(self):
        original = discovery._run
        with tempfile.TemporaryDirectory() as runner, patch.dict(os.environ, RUNNER_TEMP=runner), self.fixture_pin():
            def corrupt_disk(**kwargs):
                result = original(**kwargs)
                graph = next(Path(runner).rglob("frozen_inference_graph.pb"))
                with graph.open("r+b") as target:
                    target.write(b"!")
                return result
            with patch.object(discovery, "_run", side_effect=corrupt_disk), self.assertRaisesRegex(RuntimeError, "materialized graph identity"):
                with admission.verified_graph(Path(runner), opener=self.opener, clock=Clock()):
                    self.fail("corrupted materialized graph yielded")
            self.assertEqual(list(Path(runner).iterdir()), [])

    def test_incomplete_graph_write_fails_and_cleans(self):
        original_open = Path.open
        class ShortWriter:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                self.stream.__enter__()
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def write(self, data):
                self.stream.write(data[:-1])
                return len(data)-1
        def short_open(path, mode="r", *args, **kwargs):
            result = original_open(path, mode, *args, **kwargs)
            return ShortWriter(result) if path.name == "frozen_inference_graph.pb" and mode == "xb" else result
        with tempfile.TemporaryDirectory() as runner, patch.dict(os.environ, RUNNER_TEMP=runner), self.fixture_pin(), patch.object(Path, "open", short_open):
            with self.assertRaisesRegex(RuntimeError, "write was incomplete"):
                with admission.verified_graph(Path(runner), opener=self.opener, clock=Clock()):
                    self.fail("incomplete write yielded graph")
            self.assertEqual(list(Path(runner).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
