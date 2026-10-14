# GMDCSA24 media acquisition hold — 2026-10-14

## Acceptance defect
The upstream MIT license is identified for software; independently verified
commercial rights to identifiable participant recordings have not been
established. The branch-level source-rights registry denies both training and
evaluation. The GMDCSA24 seed direct-media helper and preparation entry point
previously proceeded to filesystem/network activity before checking rights.

## Bounded correction
Both functions now call the shared fail-closed source-rights guard before
creation of directories or media/model acquisition. Synthetic tests exercise
active, heldout and direct-helper calls, and preserve invalid-subset rejection.
No rights flags, downloads, weights, sources or settings were made permissive.
The previously saved manifest path remains unguarded and must not be treated
as accepted commercial media. Source permissions require independent clearance.

## Verification status
Four local isolated synthetic admission-order tests passed. Three fail when
the proposed guards are removed. This is not full repository CI, a legal opinion,
or a measured fall-detection accuracy improvement. Exact-head CI is pending.

## Outstanding
Update docs/PROJECT_STATE.md once the GitHub content write is permitted;
its 63 KB blob write was rejected by the integration. Finish saved-manifest
provenance denial before treating the rights gate as complete.
