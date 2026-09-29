import unittest

from analytics_lab.forensic_appearance_scale import (
    AppearanceScaleProbeConfig,
    run_appearance_scale_probe,
)


class AppearanceScaleProbeTests(unittest.TestCase):
    def test_probe_reports_bounded_runner_specific_timings(self):
        ticks = iter(
            (
                0,
                1_000_000,
                10_000_000,
                12_000_000,
                20_000_000,
                23_000_000,
            )
        )
        result = run_appearance_scale_probe(
            AppearanceScaleProbeConfig(
                candidate_count=8,
                dimensions=8,
                iterations=3,
                result_limit=5,
            ),
            clock_ns=lambda: next(ticks),
        )
        self.assertEqual(result.candidate_count, 8)
        self.assertEqual(result.dimensions, 8)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.result_limit, 5)
        self.assertEqual(result.result_count, 5)
        self.assertEqual(result.elapsed_ms, (1.0, 2.0, 3.0))
        self.assertEqual(result.median_ms, 2.0)
        self.assertEqual(result.max_ms, 3.0)
        self.assertFalse(result.identity_claim)
        self.assertFalse(result.authorizes_action)
        self.assertFalse(result.performance_claim)

    def test_result_count_is_capped_by_candidate_count(self):
        ticks = iter((0, 1_000_000))
        result = run_appearance_scale_probe(
            AppearanceScaleProbeConfig(
                candidate_count=3,
                dimensions=4,
                iterations=1,
                result_limit=100,
            ),
            clock_ns=lambda: next(ticks),
        )
        self.assertEqual(result.result_count, 3)

    def test_config_bounds_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "candidate_count"):
            AppearanceScaleProbeConfig(candidate_count=0)
        with self.assertRaisesRegex(ValueError, "candidate_count"):
            AppearanceScaleProbeConfig(candidate_count=10_001)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            AppearanceScaleProbeConfig(dimensions=1)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            AppearanceScaleProbeConfig(dimensions=513)
        with self.assertRaisesRegex(ValueError, "iterations"):
            AppearanceScaleProbeConfig(iterations=0)
        with self.assertRaisesRegex(ValueError, "iterations"):
            AppearanceScaleProbeConfig(iterations=6)
        with self.assertRaisesRegex(ValueError, "result_limit"):
            AppearanceScaleProbeConfig(result_limit=0)
        with self.assertRaisesRegex(ValueError, "result_limit"):
            AppearanceScaleProbeConfig(result_limit=1001)

    def test_invalid_clock_interval_fails_closed(self):
        ticks = iter((2_000_000, 1_000_000))
        with self.assertRaisesRegex(ValueError, "monotonic interval"):
            run_appearance_scale_probe(
                AppearanceScaleProbeConfig(
                    candidate_count=2,
                    dimensions=4,
                    iterations=1,
                    result_limit=2,
                ),
                clock_ns=lambda: next(ticks),
            )

    def test_clock_must_be_callable(self):
        with self.assertRaisesRegex(ValueError, "callable"):
            run_appearance_scale_probe(
                AppearanceScaleProbeConfig(
                    candidate_count=2,
                    dimensions=4,
                    iterations=1,
                    result_limit=2,
                ),
                clock_ns=None,  # type: ignore[arg-type]
            )

    def test_real_clock_smoke_preserves_nonclaim_semantics(self):
        result = run_appearance_scale_probe(
            AppearanceScaleProbeConfig(
                candidate_count=16,
                dimensions=8,
                iterations=1,
                result_limit=4,
            )
        )
        self.assertEqual(result.result_count, 4)
        self.assertGreaterEqual(result.elapsed_ms[0], 0.0)
        self.assertFalse(result.identity_claim)
        self.assertFalse(result.authorizes_action)
        self.assertFalse(result.performance_claim)


if __name__ == "__main__":
    unittest.main()
