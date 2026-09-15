# Project state — 2026-09-15

## Product direction
Commercially sold analytics for integration into the owner's platforms. Standalone lab; no VMS changes or automatic production releases. One recurring development task and at most one active implementation PR. Public source does not change the existing proprietary commercial-use restrictions.

## Implemented baseline
The dependency-free temporal engine consumes ordered posture observations and emits a standard v0.1 person-down **candidate** event after a configurable uninterrupted period. It bounds track state, rejects invalid/reordered inputs, resets continuity on unknown/low-confidence data or gaps, suppresses repeated events within one episode, and produces deterministic session-scoped event IDs with evidence-window references.

A new video boundary decouples file decoding from perception. `analytics_lab.video` provides strict frame/timestamp handling, a typed perception contract, bounded frame/observation/event processing, duplicate-track rejection, and a bridge into the temporal engine. `OpenCVVideoFileSource` is optional and lazy: it accepts only an existing bounded local file, rejects URLs and device paths, never downloads a model, and derives deterministic frame timestamps from an explicit caller-supplied start time plus validated FPS. Perception remains separately injected and must provide temporary session-local track IDs, posture and confidence.

This closes **local video decode -> perception contract -> temporal candidate event** as an integration boundary. It does **not** yet close detector/pose/tracker inference because no model weights have been admitted.

## Observed validation
Baseline integration commit `659ea73a9149d8bb739f019f8784aafa81870ecb` had 35 local unit tests and one deterministic observation replay. Efficiency controls were integrated at `795668874ab2598409206abde12f1f3715828a2c` after first-attempt Linux, Windows and final-gate success.

For the video-boundary work, Python 3.13.5 locally passed 39 reconstructed temporal + video tests. The video suite included an actual 16-frame MJPG AVI created and decoded through OpenCV 4.13.0, with an injected deterministic perception stub; it produced one evidence-linked candidate event at the expected 3-second boundary. This validates decoding/timestamp/bridge behavior only. It is not a detector, pose, tracking, fall-recall or real-world accuracy test. Exact-head GitHub CI remains authoritative for the repository's full suite.

## Donor review
`Tau-J/rtmlib` commit `03a1693e59e4f7cd84582c0fb30459b3bf18ad42` was reviewed as a code candidate (Apache-2.0), but its default `Body` detector path uses HumanArt-trained YOLOX weights. HumanArt's official dataset access is described for non-commercial purposes, and an OpenMMLab issue dated 2026-08-19 asks for unresolved commercial/redistribution clarification for the exact default YOLOX-M checkpoint. Therefore no RTMLib weights are admitted or downloaded. See `docs/donor-review-rtmlib.md`.

RTMLib's current `PoseTracker` also does not expose the explicit stable track-ID contract Analytics Lab requires. A reviewed tracking/association adapter is still needed.

## Efficiency control status
`AGENTS.md` and `ops/efficiency-policy.json` enforce one worker/one PR, one unchanged retry, two CI-triggering requests per session, bounded donor research, and replan/stop thresholds. Main branch protection was reported OFF during setup; CI and merge policy are not an administrator-proof lock. That known owner-side setting does not block product implementation.

## Reproduce
From the repository root with Python 3.11+:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

The OpenCV local-file path is optional and not installed by the current dependency-free CI. Production packaging must pin and audit any decoder dependency before shipping.

## Next executable step
Select and hash-pin one commercially rights-cleared detector + pose artifact pair and one explicit stable track association path. Implement that perception adapter against `analytics_lab.video.PerceptionObservation` with automatic downloads disabled. Then run one authorized annotated positive clip and normal negative clips through the complete video -> perception -> temporal path. Keep media outside the public repo and record false alerts, misses, latency and hardware settings.

## Outstanding release gates
Rights-cleared detector/pose/tracker artifacts; real model execution; artifact-level provenance register; authorized labeled golden video tests; withheld positive/negative evaluation by camera/site; documented failure cases; false-alert and missed-event rates; compute limits; security/privacy review; installable versioned adapter; owner-approved commercial release. Synthetic/stub tests do not close these gates.
