from __future__ import annotations

import unittest

from analytics_lab.tracking_benchmark_plan import TrackingBenchmarkPlan
from analytics_lab.wikimedia_tracking_frame_evidence import hash_rgb24_sequence


class TrackingFrameEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = TrackingBenchmarkPlan(
            source_id="fixture",
            source_sha256="0" * 64,
            start_frame=10,
            frame_count=2,
            fps_numerator=25,
            fps_denominator=1,
            image_width=2,
            image_height=1,
            object_class="person",
            purpose="contract regression only",
            selection_basis="synthetic bytes exercise hashing contract only",
        )
        self.frames = (
            (10, bytes([1, 2, 3, 4, 5, 6])),
            (11, bytes([7, 8, 9, 10, 11, 12])),
        )

    def test_hashes_complete_exact_window(self) -> None:
        digests, manifest = hash_rgb24_sequence(self.plan, self.frames)
        self.assertEqual([10, 11], [item.frame_index for item in digests])
        self.assertEqual(64, len(manifest))
        self.assertTrue(all(len(item.sha256) == 64 for item in digests))

    def test_hashing_is_deterministic_and_frame_bound(self) -> None:
        first, first_manifest = hash_rgb24_sequence(self.plan, self.frames)
        second, second_manifest = hash_rgb24_sequence(self.plan, self.frames)
        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)

        swapped_payloads = (
            (10, self.frames[1][1]),
            (11, self.frames[0][1]),
        )
        swapped, swapped_manifest = hash_rgb24_sequence(self.plan, swapped_payloads)
        self.assertNotEqual(first, swapped)
        self.assertNotEqual(first_manifest, swapped_manifest)

    def test_rejects_missing_reordered_or_extra_frames(self) -> None:
        with self.assertRaises(ValueError):
            hash_rgb24_sequence(self.plan, self.frames[:1])
        with self.assertRaises(ValueError):
            hash_rgb24_sequence(self.plan, tuple(reversed(self.frames)))
        with self.assertRaises(ValueError):
            hash_rgb24_sequence(self.plan, self.frames + ((12, self.frames[0][1]),))

    def test_rejects_wrong_rgb24_length(self) -> None:
        with self.assertRaises(ValueError):
            hash_rgb24_sequence(
                self.plan,
                ((10, b"short"), (11, self.frames[1][1])),
            )


if __name__ == "__main__":
    unittest.main()
