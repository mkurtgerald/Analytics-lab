# Project state — 2026-09-16

## Product direction
Commercially sold analytics for integration into the owner's platforms. Standalone lab; no VMS changes or automatic production releases. One recurring development task and at most one active implementation PR. Public source does not change the existing proprietary commercial-use restrictions.

## Implemented baseline
The dependency-free temporal engine consumes ordered posture observations and emits a standard v0.1 person-down **candidate** event after a configurable uninterrupted period. It bounds track state, rejects invalid/reordered inputs, resets continuity on uncertainty/gaps, suppresses duplicate events within one episode, and produces deterministic session-scoped evidence windows.

`analytics_lab.video` closes local file decode -> typed perception contract -> temporal candidate event. Its optional OpenCV source accepts only bounded ordinary local files, not URLs or device capture, and owns deterministic frame timestamps.

`analytics_lab.perception` defines validated boxes/keypoints/pose candidates, deterministic temporary IoU association, conservative pose-to-posture geometry with explicit abstention, and a `PosePerceptionAdapter` that emits existing `PerceptionObservation` values. Track labels are session-local only; there is no biometric identity or ReID.

`analytics_lab.artifacts` provides fail-closed, local-only model artifact admission by exact path, byte count and SHA-384. The recorded manifest baseline is Open Model Zoo FP16 `person-detection-retail-0013` + `human-pose-estimation-0001` at upstream commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`.

`analytics_lab.openvino_omz` is an optional OpenVINO 2026.3.1 adapter boundary. It verifies the exact four local model artifacts before runtime creation, parses the reviewed SSD detector output, crops detected people, runs the reviewed pose model, extracts only the four shoulder/hip heatmap peaks required by the posture classifier, and emits `PoseCandidate` values. The keypoint channel IDs and quarter-pixel peak refinement are adapted from the Apache-2.0 Open Model Zoo OpenPose decoder at the same pinned commit. The full PAF grouping decoder is intentionally not copied: the detector-isolated crop method is a small integration baseline that must earn its accuracy through labeled-video measurement.

`analytics_lab.openvino_pipeline` composes verified OMZ artifacts -> reviewed runtime backend -> pose candidates -> deterministic temporary tracks -> posture observations -> the existing temporal person-down candidate engine. It exposes both an already-decoded frame-source path and a bounded ordinary-local-video path, preserving the existing no-URL/no-device/no-downloader controls. Runtime version and device identity are carried in the returned result so evaluation evidence can bind to the executed runtime lineage.

`analytics_lab.evaluation` provides deterministic one-to-one matching of person-down candidate evidence windows against bounded labeled person-down intervals and bounded multi-sample aggregation across sites/cameras. It reports matched alerts, misses, false alerts, precision/recall when denominators exist, false alerts per camera-hour, alert delay, and deterministic site/camera summaries. Duplicate alerts cannot inflate recall; duplicate sample IDs and overlapping intervals for the same site-scoped camera are rejected so repeated footage cannot silently inflate metrics. Camera summaries carry explicit site and camera identity so common local camera IDs can be reused safely at different sites.

`analytics_lab.validation` provides the execution boundary needed for reproducible real-video measurement. Each sample must provide an opaque rights/authorization reference, ordinary local non-symlink video path, expected SHA-256 media identity, bounded site/camera/sample IDs, a concrete evaluation interval and labels. All sample paths, checksums, byte budget, duplicate IDs and same-site/same-camera interval overlaps are validated before inference begins. The suite then runs the existing reviewed local-video OpenVINO path, requires one comparable runtime version and CPU device across the suite, records per-sample/aggregate wall-clock throughput, binds the four reviewed model artifact identities, and feeds resulting candidate events into the existing multi-site evaluation layer. It returns metrics and provenance identities only; it does not retain media.

The validation CLI provides a bounded machine-readable entry point for that suite. `python -m analytics_lab.validation_cli --manifest <local-json>` accepts only a strict versioned local JSON manifest, rejects unknown fields and network/device paths, constructs the existing rights-bound sample/config objects, runs the existing suite, and emits a compact JSON evidence document containing rights references, media/model/runtime/device identities and measurements but no media/model source paths or frame content. It does not add downloaders, network capture or release automation.

## Observed validation
Baseline integration commit `659ea73a9149d8bb739f019f8784aafa81870ecb` had 35 local unit tests. Efficiency controls were integrated at `795668874ab2598409206abde12f1f3715828a2c`. Local-video bridge commit `7f72435937eca7363bc35b5788b878192c607843` passed exact-head PR CI and first-attempt post-merge CI. Pose/artifact commit `c5aa7c05980f7145e8bf5cb379ff5f5da5133796` passed exact-head PR CI and first-attempt post-merge Linux/Windows/quality-gate CI. OpenVINO adapter commit `7c56e56888560e6d9c7e0399ab74d958af117516` passed exact-head PR CI and first-attempt post-merge Linux/Windows/quality-gate CI. Pipeline composition commit `b8937687a8a0586066ce7d0b07be9da963a5f6e5` passed its post-merge Linux, Windows and Analytics quality gate checks on the first attempt. Evaluation commit `104f832364e4e759ecbdabbd45667e2d4ef3f141` passed its exact-head and post-merge Linux, Windows and Analytics quality gate checks on the first attempt. Multi-sample evaluation commit `0020f2ed7c3fd89a63f08e5bd2a554b4b12c0324` passed its exact-head and post-merge Linux, Windows and Analytics quality gate checks on the first attempt. Validation-suite commit `6ffbe13361a0096270b59059cea87d751c719fe6` passed its exact-head and post-merge Linux, Windows and Analytics quality gate checks on the first attempt. Validation-CLI commit `48b44e97e110811dfcec52e996fe8075a4b2d011` passed its exact-head and first-attempt post-merge Linux, Windows and Analytics quality gate checks.

A site/camera scoping regression was reproduced locally: the prior camera-only grouping rejected overlapping time ranges for two different sites that both used a local ID such as `cam-01`. The current candidate scopes overlap checks and camera counts by `(site_id, camera_id)` while preserving rejection of real overlap within one site/camera. Focused deterministic regression passed before repository mutation; exact-head GitHub CI remains authoritative for the full suite.

No detector, pose, fall, person-down or tracking accuracy claim is made by structural, injected-runtime, synthetic evaluation, aggregation, validation-orchestration or CLI tests.

## Donor review
RTMLib commit `03a1693e59e4f7cd84582c0fb30459b3bf18ad42` remains a code-only candidate; its default HumanArt-trained detector weights are hold/do-not-ship because artifact-specific commercial/redistribution rights remain unresolved in reviewed evidence.

OpenVINO Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e` remains the selected artifact baseline. The person detector is documented as BGR `1x3x320x544` with SSD rows `1x1x200x7`; the pose model is documented as BGR `1x3x256x456` with 38-channel PAF and 19-channel heatmap outputs. Model files remain local-provision-only.

OpenVINO Runtime tag `2026.3.1` resolves to commit `759c5a6ab8c066af5f4bc5ebd04643706012a37d`; upstream source and the current PyPI release identify Apache-2.0. The adapter requires a runtime version string beginning with `2026.3.1`. Platform-specific wheel hashes, bundled/native dependency notices and final product redistribution packaging are still open release items; the runtime package is not committed to this repository.

## Efficiency control status
`AGENTS.md` and `ops/efficiency-policy.json` enforce one worker/one PR, one unchanged retry, two CI-triggering requests per session, bounded donor research, and replan/stop thresholds. Main branch protection was reported OFF during setup; the known owner-side setting does not block product implementation.

## Reproduce
From the repository root with Python 3.11+:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

With already provisioned reviewed artifacts/runtime plus an authorized local manifest:

```sh
python -m analytics_lab.validation_cli --manifest /path/to/local-validation.json
```

The OpenCV local-file path and OpenVINO model runtime remain optional. Product packaging must pin and audit decoder/runtime dependencies before shipping.

## Next executable step
Provision the already recorded model artifacts plus OpenVINO 2026.3.1 outside the public repository and perform one bounded CPU validation-suite run using the manifest CLI. The worker environment still does not have OpenVINO installed, and no large runtime download is authorized by the efficiency policy, so no runtime package is fetched here. Use only authorized labeled positive/negative local clips with an opaque rights reference and pre-recorded SHA-256. The suite should produce exact media/model/runtime/device identities, candidate metrics, per-site/per-camera results and measured end-to-end throughput without retaining media. If the detector-isolated four-keypoint baseline underperforms on prone-person recall, compare it against the pinned full OpenPose decoder before adding another framework.

## Outstanding release gates
Actual rights-reviewed runtime execution; platform wheel/native dependency provenance; authorized labeled golden video tests; withheld positive/negative evaluation by camera/site; documented failure cases; false-alert and missed-event rates; compute limits; security/privacy review; packaging/notices; installable versioned adapter; owner-approved commercial release. Synthetic/stub geometry and evaluation tests do not close these gates.
