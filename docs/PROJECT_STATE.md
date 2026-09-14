# Project state — 2026-09-14

## Product direction
Commercially sold analytics for integration into the owner's platforms. Standalone lab; no VMS changes or automatic production releases. One recurring development task; one active implementation PR. Public source does not change the existing proprietary commercial-use restrictions.

## Implemented baseline
A dependency-free Python temporal engine consumes ordered posture observations from a future perception adapter and emits a standard v0.1 person-down **candidate** event after a configurable uninterrupted period. It bounds track state, rejects invalid/reordered inputs, resets continuity on unknown/low-confidence data or gaps, suppresses repeated events within one continuous episode, and produces deterministic session-scoped event IDs with observation-window references.

The observation CLI has bounded input lines, JSON-only output and nonzero errors without echoing untrusted input. It does not open video, connect to webcams, download models, record footage or perform medical/causal inference. Replay can emit valid prefix events before a later input error; consumers must inspect the exit code. Start a new engine/session when a stream or tracker resets. Track IDs are temporary within-camera labels, not identities. The engine is synchronous, not thread-safe; it does not persist state across restarts.

## Observed validation
Local Python 3.13.5: 35 unittest cases passed. Synthetic replay produced exactly one candidate event. No video was analyzed and no accuracy, fall-detection recall, latency-under-video-load, or commercial-readiness claim is made.

GitHub CI is configured to run these tests on standard Ubuntu and Windows hosted runners with Python 3.11. Actions are commit-pinned and use read-only permissions; no model installation or credentials are needed. Check live PR checks before claiming CI passed. Do not use self-hosted/physical-camera runners.

## Reproduce
From the repository root with Python 3.11+:

```sh
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Input fields: integer UTC `timestamp_ms`, opaque `track_id`, `posture` (`upright`, `down`, `other`, `unknown`), and finite `confidence` in [0,1]. The sample's timestamps start at the Unix epoch solely for deterministic synthetic replay. Real adapters must distinguish event time from processing time and session changes. Current event confidence is the minimum supporting input score, explicitly not a calibrated event probability. Threshold defaults are engineering fixtures, not validated operating points.

## Next executable step
Review and pin one RTMLib/RTMPose detector/pose path and its exact model artifacts and dependencies for commercial integration. Then add an optional local-video adapter with stable per-person tracks, timestamp handling, explicit unknown observations and no automatic model download. Use an authorized annotated clip; keep media outside the public repo. Compare emitted observations and events against labels. Do not train a detector or add several perception frameworks at once.

## Outstanding release gates
Video ingestion and model execution; artifact-level license/provenance register; end-to-end golden video tests; withheld positive and negative evaluation by camera/site; documented failure cases; false-alert and missed-event rates; compute limits; security/privacy review; installable versioned adapter; owner-approved commercial release. None are marked complete by the synthetic unit tests.
