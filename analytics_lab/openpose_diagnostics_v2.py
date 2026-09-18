"""Posture-quality variant of the evidence-only OpenPose diagnostic."""
from __future__ import annotations

from .posture_quality_diagnostics import main, run_posture_quality_diagnostic

# Preserve the historical callable name used by tests/tools while routing the
# evidence lane through the current acceptance-moving diagnostic.
run_reference_diagnostic = run_posture_quality_diagnostic


if __name__ == "__main__":
    raise SystemExit(main())
