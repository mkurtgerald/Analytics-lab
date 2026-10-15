"""Commercial-rights preflight must precede all diagnostic side effects."""
from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class FigshareGeneralizationPreflightTests(unittest.TestCase):
    def test_denied_rights_prevent_disk_media_and_model_work(self) -> None:
        for variant in (2, 3, 4, 5):
            with self.subTest(generalization=variant), tempfile.TemporaryDirectory() as tmp:
                module = importlib.import_module(
                    f"analytics_lab.figshare_generalization{variant}_person_down_diagnostic"
                )
                output = Path(tmp) / "must-not-be-created"
                candidate = Path(tmp) / "models"

                def forbidden_opener(*args, **kwargs):
                    self.fail("denied rights must not initiate network requests")

                with (
                    mock.patch.object(
                        module, "require_source",
                        side_effect=ValueError("commercial authorization missing"),
                    ) as rights,
                    mock.patch.object(
                        module, "admit_pinned_pair", side_effect=AssertionError("media admitted")
                    ) as media,
                    mock.patch.object(
                        module, "_download_models", side_effect=AssertionError("models downloaded")
                    ) as models,
                    mock.patch.object(
                        module, "_provision", side_effect=AssertionError("candidate provisioned")
                    ) as provision,
                ):
                    with self.assertRaisesRegex(ValueError, "commercial authorization missing"):
                        getattr(
                            module,
                            f"run_figshare_generalization{variant}_person_down_diagnostic",
                        )(output, candidate, opener=forbidden_opener)

                rights.assert_called_once_with(
                    module.FIGSHARE_FALL_2017.source_id,
                    purpose="evaluation",
                    require_real_world=True,
                )
                media.assert_not_called()
                models.assert_not_called()
                provision.assert_not_called()
                self.assertFalse(output.exists())
                self.assertFalse(candidate.exists())


if __name__ == "__main__":
    unittest.main()
