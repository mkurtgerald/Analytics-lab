import hashlib
from pathlib import Path
import tempfile
import unittest

from analytics_lab.artifacts import ArtifactSpec, OPENVINO_OMZ_2023_FP16, verify_artifact_set, verify_local_artifact
from analytics_lab.perception import (
    BBox, IoUTracker, Keypoint, PoseCandidate, PosePerceptionAdapter,
    PostureConfig, TrackerConfig, classify_posture,
)


def pose(box, orientation="upright", confidence=0.9):
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    if orientation == "upright":
        pts = (
            Keypoint("left_shoulder", x1 + box.width * .35, y1 + box.height * .25, confidence),
            Keypoint("right_shoulder", x1 + box.width * .65, y1 + box.height * .25, confidence),
            Keypoint("left_hip", x1 + box.width * .40, y1 + box.height * .65, confidence),
            Keypoint("right_hip", x1 + box.width * .60, y1 + box.height * .65, confidence),
        )
    elif orientation == "down":
        pts = (
            Keypoint("left_shoulder", x1 + box.width * .25, y1 + box.height * .35, confidence),
            Keypoint("right_shoulder", x1 + box.width * .25, y1 + box.height * .65, confidence),
            Keypoint("left_hip", x1 + box.width * .70, y1 + box.height * .40, confidence),
            Keypoint("right_hip", x1 + box.width * .70, y1 + box.height * .60, confidence),
        )
    else:
        pts = (
            Keypoint("left_shoulder", x1 + box.width * .25, y1 + box.height * .25, confidence),
            Keypoint("right_shoulder", x1 + box.width * .45, y1 + box.height * .25, confidence),
            Keypoint("left_hip", x1 + box.width * .55, y1 + box.height * .60, confidence),
            Keypoint("right_hip", x1 + box.width * .75, y1 + box.height * .60, confidence),
        )
    return PoseCandidate(box, pts, confidence)


class ArtifactTests(unittest.TestCase):
    def make_spec(self, data=b"artifact"):
        return ArtifactSpec("test", "models/a.bin", len(data), hashlib.sha384(data).hexdigest(),
                            "https://example.com/a.bin", "Apache-2.0", "https://example.com/license")

    def test_verified_artifact(self):
        data = b"artifact"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "models/a.bin")
            path.parent.mkdir()
            path.write_bytes(data)
            result = verify_local_artifact(tmp, self.make_spec(data), chunk_bytes=2)
            self.assertEqual(result.path, path.resolve())

    def test_size_and_checksum_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "models/a.bin")
            path.parent.mkdir()
            path.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                verify_local_artifact(tmp, self.make_spec(b"artifact"))
            bad = ArtifactSpec("test", "models/a.bin", 5, hashlib.sha384(b"other").hexdigest(),
                               "https://example.com/a.bin", "Apache-2.0", "https://example.com/license")
            with self.assertRaises(ValueError):
                verify_local_artifact(tmp, bad)

    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp, "target")
            target.write_bytes(b"artifact")
            link = Path(tmp, "models/a.bin")
            link.parent.mkdir()
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with self.assertRaises(ValueError):
                verify_local_artifact(tmp, self.make_spec())

    def test_duplicate_artifacts_fail(self):
        spec = self.make_spec()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "models/a.bin")
            path.parent.mkdir()
            path.write_bytes(b"artifact")
            with self.assertRaises(ValueError):
                verify_artifact_set(tmp, [spec, spec])

    def test_unsafe_artifact_paths_fail(self):
        digest = hashlib.sha384(b"x").hexdigest()
        for name in ["../a.bin", "/a.bin", "models//a.bin", "models\\a.bin", "C:/a.bin"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                ArtifactSpec("test", name, 1, digest, "https://example.com/a",
                             "Apache-2.0", "https://example.com/license")

    def test_omz_manifest_is_hash_pinned(self):
        self.assertEqual(len(OPENVINO_OMZ_2023_FP16), 4)
        self.assertEqual({x.license_id for x in OPENVINO_OMZ_2023_FP16}, {"Apache-2.0"})
        self.assertTrue(all(len(x.sha384) == 96 for x in OPENVINO_OMZ_2023_FP16))
        self.assertTrue(all("6697dead54ed1cdd664b0313189c2cb52ee6335e" in x.license_url
                            for x in OPENVINO_OMZ_2023_FP16))


class TrackerTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(BBox(0, 0, 10, 10).iou(BBox(5, 0, 15, 10)), 1 / 3)
        self.assertEqual(BBox(0, 0, 1, 1).iou(BBox(2, 2, 3, 3)), 0)

    def test_stable_track_across_motion_and_one_missing_frame(self):
        tracker = IoUTracker(TrackerConfig(min_iou=.2, max_missed_frames=1))
        first = tracker.update(0, (BBox(0, 0, 10, 20),))
        tracker.update(1, ())
        third = tracker.update(2, (BBox(1, 0, 11, 20),))
        self.assertEqual(first, third)

    def test_track_expires_and_id_is_not_reused(self):
        tracker = IoUTracker(TrackerConfig(min_iou=.2, max_missed_frames=0))
        first = tracker.update(0, (BBox(0, 0, 10, 20),))[0]
        tracker.update(1, ())
        second = tracker.update(2, (BBox(0, 0, 10, 20),))[0]
        self.assertNotEqual(first, second)
        self.assertEqual(second, "track-000002")

    def test_two_people_keep_ids_when_backend_order_changes(self):
        tracker = IoUTracker(TrackerConfig(min_iou=.2))
        left = BBox(0, 0, 10, 20)
        right = BBox(100, 0, 110, 20)
        first = tracker.update(0, (left, right))
        swapped = tracker.update(1, (BBox(101, 0, 111, 20), BBox(1, 0, 11, 20)))
        self.assertEqual(swapped, (first[1], first[0]))

    def test_max_cardinality_matching_avoids_avoidable_track_fragmentation(self):
        tracker = IoUTracker(TrackerConfig(min_iou=.2))
        first = tracker.update(0, (BBox(0, 0, 10, 10), BBox(6, 0, 16, 10)))

        # The first new box overlaps both existing tracks, while the second box
        # overlaps only the first track above threshold. Greedy highest-IoU
        # assignment takes the shared box for the first track and creates an
        # unnecessary third track. Maximum-cardinality association preserves both.
        second = tracker.update(1, (BBox(2, 0, 12, 10), BBox(-3, 0, 7, 10)))

        self.assertEqual(second, (first[1], first[0]))
        self.assertEqual(tracker.active_tracks, 2)

    def test_capacity_and_order_fail_closed(self):
        tracker = IoUTracker(TrackerConfig(max_tracks=1))
        tracker.update(0, (BBox(0, 0, 10, 20),))
        with self.assertRaises(ValueError):
            tracker.update(0, ())
        with self.assertRaises(RuntimeError):
            tracker.update(1, (BBox(0, 0, 10, 20), BBox(100, 0, 110, 20)))
        kept = tracker.update(1, (BBox(1, 0, 11, 20),))
        self.assertEqual(kept, ("track-000001",))
        with self.assertRaises(ValueError):
            tracker.update(1, ())


class PostureTests(unittest.TestCase):
    def test_upright(self):
        result = classify_posture(pose(BBox(0, 0, 40, 120), "upright"))
        self.assertEqual(result.posture, "upright")
        self.assertEqual(result.basis, "vertical_torso_and_tall_bbox")

    def test_down(self):
        result = classify_posture(pose(BBox(0, 0, 120, 40), "down"))
        self.assertEqual(result.posture, "down")
        self.assertEqual(result.basis, "horizontal_torso_and_wide_bbox")

    def test_ambiguous_is_other(self):
        result = classify_posture(pose(BBox(0, 0, 80, 80), "other"))
        self.assertEqual(result.posture, "other")

    def test_missing_and_low_confidence_are_unknown(self):
        candidate = PoseCandidate(BBox(0, 0, 40, 120), (), .9)
        self.assertEqual(classify_posture(candidate).posture, "unknown")
        self.assertEqual(classify_posture(pose(BBox(0, 0, 40, 120), "upright", .2)).posture, "unknown")

    def test_collapsed_torso_is_unknown(self):
        points = tuple(Keypoint(name, 10, 10, .9) for name in
                       ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
        candidate = PoseCandidate(BBox(0, 0, 40, 120), points, .9)
        self.assertEqual(classify_posture(candidate).posture, "unknown")

    def test_threshold_config_is_validated(self):
        with self.assertRaises(ValueError):
            PostureConfig(orientation_threshold=.5)


class AdapterTests(unittest.TestCase):
    def test_adapter_emits_stable_temporary_track_and_posture(self):
        frames = {
            0: [pose(BBox(0, 0, 40, 120), "upright")],
            1: [pose(BBox(1, 0, 41, 120), "upright")],
            2: [pose(BBox(0, 0, 120, 40), "down")],
        }
        adapter = PosePerceptionAdapter(lambda image, frame, timestamp: frames[frame])
        first = adapter(object(), 0, 0)
        second = adapter(object(), 1, 100)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(first[0].posture, "upright")
        third = adapter(object(), 2, 200)
        self.assertEqual(third[0].track_id, first[0].track_id)
        self.assertEqual(third[0].posture, "down")

    def test_backend_contract_and_bound(self):
        with self.assertRaises(ValueError):
            PosePerceptionAdapter(lambda *_: None)(object(), 0, 0)
        adapter = PosePerceptionAdapter(lambda *_: [pose(BBox(i * 20, 0, i * 20 + 10, 20)) for i in range(257)])
        with self.assertRaises(RuntimeError):
            adapter(object(), 0, 0)


if __name__ == "__main__":
    unittest.main()
