"""Corrected pose-geometry variant of the evidence-only OpenPose diagnostic."""
from __future__ import annotations

from . import openpose_diagnostics as _base
from .openpose_pose_geometry import select_reference_pose as _select_reference_pose

# The base diagnostic owns the bounded media/runtime loop. Override only the
# measured association/geometry seam exposed by its first exact-head run.
_base.select_reference_pose = _select_reference_pose

main = _base.main
run_reference_diagnostic = _base.run_reference_diagnostic


if __name__ == "__main__":
    raise SystemExit(main())
