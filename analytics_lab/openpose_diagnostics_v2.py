"""Association-geometry variant of the evidence-only OpenPose diagnostic."""
from __future__ import annotations

from .openpose_association_diagnostics import main, run_association_diagnostic

# Preserve the historical callable name used by tests/tools while routing the
# evidence lane through the current acceptance-moving diagnostic.
run_reference_diagnostic = run_association_diagnostic


if __name__ == "__main__":
    raise SystemExit(main())
