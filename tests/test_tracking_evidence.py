import hashlib
import unittest

from analytics_lab.tracking_evidence import (
    EXHAUSTIVE_MULTI_OBJECT,
    FrameDigest,
    TrackingEvidenceManifest,
    canonical_frame_manifest_sha256,
    require_multi_object_tracking_evidence,
)


class TrackingEvidenceTests(unittest.TestCase):
    def test_canonical_manifest_binds_ordered_frame_hashes(self):
        first = "0" * 64
        second = "1" * 64
        expected = hashlib.sha256(f"0\t{first}\n2\t{second}\n".encode("ascii")).hexdigest()
        self.assertEqual(
            canonical_frame_manifest_sha256((FrameDigest(0, first), FrameDigest(2, second))),
            expected,
        )

    def test_frame_manifest_rejects_reordering_duplicate_or_empty(self):
        digest = "a" * 64
        with self.assertRaisesRegex(ValueError, "at least one"):
            canonical_frame_manifest_sha256(())
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            canonical_frame_manifest_sha256((FrameDigest(2, digest), FrameDigest(1, digest)))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            canonical_frame_manifest_sha256((FrameDigest(1, digest), FrameDigest(1, digest)))

    def test_manifest_rejects_single_target_dataset_for_multi_object_scoring(self):
        with self.assertRaisesRegex(ValueError, "exhaustive_multi_object"):
            self._manifest(annotation_scope="single_target")

    def test_manifest_rejects_unbound_or_unbounded_evidence(self):
        with self.assertRaisesRegex(ValueError, "annotation_sha256"):
            self._manifest(annotation_sha256="not-a-digest")
        with self.assertRaisesRegex(RuntimeError, "frame_count"):
            self._manifest(frame_count=100_001)
        with self.assertRaisesRegex(RuntimeError, "image dimensions"):
            self._manifest(image_width=40_000)

    def test_exhaustive_bound_manifest_is_eligible(self):
        manifest = self._manifest()
        self.assertIsNone(require_multi_object_tracking_evidence(manifest))

    @staticmethod
    def _manifest(**overrides):
        values = {
            "dataset": "example MOT evidence",
            "provenance": "https://example.invalid/dataset@exact-revision",
            "dataset_version": "revision-1",
            "license_expression": "CC-BY-4.0",
            "attribution": "Example Dataset Authors",
            "sequence_id": "test/sequence_450",
            "annotation_scope": EXHAUSTIVE_MULTI_OBJECT,
            "annotation_sha256": "2" * 64,
            "frame_manifest_sha256": "3" * 64,
            "frame_count": 90,
            "image_width": 1920,
            "image_height": 1080,
        }
        values.update(overrides)
        return TrackingEvidenceManifest(**values)


if __name__ == "__main__":
    unittest.main()
