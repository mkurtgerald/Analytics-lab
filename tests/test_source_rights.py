import unittest

from analytics_lab.source_rights import (
    DataSourceRights,
    FIGSHARE_FALL_2017,
    GMDCSA24,
    UE4_FALL,
    require_source,
)


class SourceRightsTests(unittest.TestCase):
    def test_synthetic_source_is_commercially_eligible_but_not_real_world_evidence(self):
        self.assertTrue(require_source(UE4_FALL.source_id, purpose="training").permits("evaluation"))
        with self.assertRaises(ValueError):
            require_source(UE4_FALL.source_id, purpose="evaluation", require_real_world=True)

    def test_real_world_figshare_source_is_eligible_for_commercial_evaluation(self):
        source = require_source(
            FIGSHARE_FALL_2017.source_id,
            purpose="evaluation",
            require_real_world=True,
        )
        self.assertEqual(source.media_origin, "real_world")
        self.assertEqual(source.license_id, "CC-BY-4.0")
        self.assertTrue(source.exact_asset_identity_required)

    def test_gmdcsa24_is_pinned_real_world_commercial_source(self):
        source = require_source(
            GMDCSA24.source_id,
            purpose="evaluation",
            require_real_world=True,
        )
        self.assertEqual(source.version, "git:5abac7693229900cf80f722e878fbb119211fc1c")
        self.assertEqual(source.media_origin, "real_world")
        self.assertEqual(source.license_id, "MIT")
        self.assertTrue(source.commercial_training)
        self.assertTrue(source.exact_asset_identity_required)

    def test_unknown_source_and_purpose_fail_closed(self):
        with self.assertRaises(ValueError):
            require_source("unknown", purpose="training")
        with self.assertRaises(ValueError):
            UE4_FALL.permits("distribution")

    def test_invalid_record_rejected(self):
        fields = dict(
            source_id="x",
            title="x",
            version="1",
            media_origin="unknown",
            license_id="MIT",
            source_url="https://example.com/x",
            license_url="https://example.com/license",
            provenance_ref="x",
            commercial_training=True,
            commercial_evaluation=True,
            attribution_required=False,
        )
        with self.assertRaises(ValueError):
            DataSourceRights(**fields)


if __name__ == "__main__":
    unittest.main()
