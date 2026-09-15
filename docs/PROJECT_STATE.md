# Project state — 2026-09-15

## Product direction
Commercially sold analytics for integration into the owner's platforms. Standalone lab; no VMS changes or automatic production releases. One recurring development task and at most one active implementation PR. Public source does not change the existing proprietary commercial-use restrictions.

## Implemented baseline
The dependency-free temporal engine consumes ordered posture observations and emits a standard v0.1 person-down **candidate** event after a configurable uninterrupted period. It bounds track state, rejects invalid/reordered inputs, resets continuity on uncertainty/gaps, suppresses duplicate events within one episode, and produces deterministic session-scoped evidence windows.

`analytics_lab.video` closes local file decode -> typed perception contract -> temporal candidate event. Its optional OpenCV source accepts only bounded ordinary local files, not URLs or device capture, and owns deterministic frame timestamps.

The current perception increment adds two model-independent boundaries:

1. `analytics_lab.perception` defines validated boxes/keypoints/pose candidates, deterministic temporary IoU association, conservative pose-to-posture geometry with explicit abstention, and a `PosePerceptionAdapter` that emits existing `PerceptionObservation` values. Track labels are session-local only; there is no biometric identity or ReID.
2. `analytics_lab.artifacts` provides fail-closed, local-only model artifact admission by exact path, byte count and SHA-384. It contains no downloader. The first recorded manifest baseline is Open Model Zoo FP16 `person-detection-retail-0013` + `human-pose-estimation-0001` at upstream commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`.

This closes **pose candidate -> temporary stable track -> posture observation** and **pinned local artifact -> verified bytes** as reusable integration boundaries. Real detector/pose inference is still not implemented.

## Observed validation
Baseline integration commit `659ea73a9149d8bb739f019f8784aafa81870ecb` had 35 local unit tests. Efficiency controls were integrated at `795668874ab2598409206abde12f1f3715828a2c`. Local-video bridge commit `7f72435937eca7363bc35b5788b878192c607843` passed exact-head PR CI and first-attempt post-merge CI; its local video suite included an actual 16-frame MJPG decode with deterministic injected observations.

For this perception/artifact increment, Python 3.13.5 locally passed 19 focused tests after fixing two deterministic test/design defects before any push: the model-manifest checksum format was correctly treated as SHA-384 rather than SHA-512, and tracker capacity handling was made atomic so a failed frame cannot partially mutate association state. Tests cover hash/size mismatch, unsafe paths/symlinks, exact donor pins, IoU association, expiry/no-ID-reuse, backend-order changes, upright->horizontal continuity, capacity/order failures, posture abstention, and adapter bounds. This focused harness reconstructs only the new boundary; exact-head GitHub CI is required for the full repository suite before merge.

No model binary was downloaded or executed. These tests do not establish real-video detector, pose, tracking, fall, or person-down accuracy.

## Donor review
RTMLib commit `03a1693e59e4f7cd84582c0fb30459b3bf18ad42` remains a code-only candidate; its default HumanArt-trained detector weights are hold/do-not-ship because artifact-specific commercial/redistribution rights remain unresolved in reviewed evidence.

OpenVINO Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e` is the selected artifact-manifest baseline for the next inference adapter. The two Intel model manifests and repository license identify Apache-2.0, and the upstream downloader implementation confirms the recorded manifest hashes are SHA-384. Exact selected files, sizes and hashes are in `analytics_lab.artifacts` and `docs/donor-review-openvino-omz.md`. Model files remain local-provision-only; product packaging/redistribution and OpenVINO Runtime dependency admission are not yet closed.

## Efficiency control status
`AGENTS.md` and `ops/efficiency-policy.json` enforce one worker/one PR, one unchanged retry, two CI-triggering requests per session, bounded donor research, and replan/stop thresholds. Main branch protection was reported OFF during setup; the known owner-side setting does not block product implementation.

## Reproduce
From the repository root with Python 3.11+:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

The OpenCV local-file path and future model runtime remain optional. Production packaging must pin and audit decoder/runtime dependencies before shipping.

## Next executable step
Implement one optional OpenVINO inference backend that consumes only successfully verified local detector/pose artifacts and never downloads at runtime. Translate detector/pose output into `PoseCandidate` values and preserve the temporary-track boundary. Reuse/adapt reviewed permissive Open Model Zoo post-processing rather than rebuilding it when practical. Then run one authorized annotated positive clip plus normal negatives through video -> perception -> temporal event and record false alerts, misses, latency and hardware settings.

## Outstanding release gates
Actual rights-reviewed runtime execution; authorized labeled golden video tests; withheld positive/negative evaluation by camera/site; documented failure cases; false-alert and missed-event rates; compute limits; security/privacy review; packaging/notices; installable versioned adapter; owner-approved commercial release. Synthetic/stub geometry tests do not close these gates.
