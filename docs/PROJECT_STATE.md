# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed Open Model Zoo person detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent.

`analytics_lab.video` owns bounded ordinary-local-file decode and timestamps. `analytics_lab.perception` owns validated boxes/keypoints, temporary session-local tracks and explicit posture abstention. `analytics_lab.openvino_omz` owns exact artifact verification plus the reviewed OpenVINO/OMZ detector-pose adapter. `analytics_lab.openvino_pipeline` composes perception with fresh tracking/temporal state. `analytics_lab.evaluation` measures one-to-one matches, misses, false alerts, false alerts per camera-hour and alert delay across site-scoped cameras. `analytics_lab.validation` binds rights references, media SHA-256 identities, model/runtime/device identities and throughput evidence without retaining video. `analytics_lab.validation_cli` exposes that suite through a strict local JSON manifest.

## Current acceptance-moving change
Local video decode now derives frame timing from the media/container clock exposed by OpenCV (`CAP_PROP_POS_MSEC`) when that clock is available, anchored so the first decoded frame still equals the manifest's declared start timestamp. The previous implementation always synthesized timestamps from `frame_index / average_fps`, which can distort event persistence, alert-delay, decoded-duration, and camera-hour evidence for variable-frame-rate or irregularly timed footage.

A focused deterministic harness covered three cases before repository mutation: variable media timestamps preserve their irregular spacing, a stalled media clock falls back to nominal FPS without moving time backwards, and non-zero source PTS is re-anchored to the declared sample start. Repository regressions use an injected fake OpenCV capture so this timing behavior is tested without downloading or retaining media. This change affects timestamp fidelity only; it does not establish detector, pose, tracking, or person-down accuracy.

The previously merged validation protections remain in place: temporary IoU association uses maximum-cardinality matching; validation-manifest paths reject network/device namespaces; manifest-relative paths resolve against the manifest directory; evaluation matching maximizes one-to-one cardinality; validation duration is bound to decoded timestamp coverage; multi-sample validation prepares the reviewed OpenVINO backend once per suite while each sample gets fresh perception/tracker/temporal state.

## Reviewed donor/provenance baseline
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`.
- Pinned FP16 artifacts: `person-detection-retail-0013` and `human-pose-estimation-0001`, admitted only by exact size/SHA-384 through the local artifact manifest.
- OpenVINO Runtime release baseline: tag `2026.3.1`, commit `759c5a6ab8c066af5f4bc5ebd04643706012a37d`, Apache-2.0 source lineage. Platform wheel hashes, native dependency notices and final redistribution packaging remain release gates.
- RTMLib commit `03a1693e59e4f7cd84582c0fb30459b3bf18ad42` remains code-only; its default HumanArt-trained detector weights stay hold/do-not-ship because commercial/redistribution rights remain unresolved in reviewed evidence.

## Verified repository baseline
Main before this work item: `65e2b95972983bab94b5ca199fdeffffee99d48a`. Its post-merge Analytics quality workflow completed successfully. No implementation PR was open at intake and no repository workflow run was queued or running.

Before mutation, `tools/guardrails.py preflight` was run with fresh verified counts and allowed implementation. A focused local timestamp-selection harness passed 3/3 cases: variable media timing, stalled-clock fallback, and non-zero PTS anchoring. Exact-head GitHub Linux, Windows and Analytics quality-gate checks remain authoritative before merge.

## Efficiency / execution ledger
One worker, one active acceptance-moving work item, at most one implementation PR. Current work item: preserve decoded media timing for real-video validation evidence. Base: `65e2b95972983bab94b5ca199fdeffffee99d48a`. Open implementation PRs at intake: 0. Active runs for the base head before mutation: 0. Unchanged retries: 0. CI-triggering requests before opening the implementation PR: 0. Consecutive sessions without tested acceptance improvement: 0.

## Reproduce
Dependency-free repository checks:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

With externally provisioned reviewed artifacts/runtime and an authorized local manifest:

```sh
python -m analytics_lab.validation_cli --manifest /path/to/local-validation.json
```

Relative `artifact_root` and `video_path` entries in that manifest are interpreted relative to the manifest file itself. Network/UNC/device namespace paths are rejected before validation media or model paths are touched. Local decode uses the media/container timestamp clock when available, with a bounded nominal-FPS fallback only when the media clock cannot advance monotonically.

## Next executable step
Run one bounded CPU validation suite with externally provisioned OpenVINO `2026.3.1`, the exact pinned OMZ artifacts, and authorized labeled positive/negative local clips. Record exact media/model/runtime/device identities, preparation time, decoded evaluation duration, processing throughput, misses, false alerts per camera-hour, alert delay and per-site/per-camera results. Media-derived frame timestamps should now make persistence/latency/duration evidence resilient to variable or irregular source timing. Use resulting multi-person failure cases to measure whether the improved temporary-track continuity reduces missed persistence events before changing tracker thresholds or adding another tracking framework. If prone-person recall is inadequate, compare the current detector-isolated four-keypoint method with the pinned full OpenPose decoder before adding another framework.

## Outstanding commercial-release gates
Actual rights-reviewed runtime execution; platform wheel/native dependency provenance; authorized held-out positive and normal-negative video; false-alert and missed-event measurements; documented failure cases; comparable latency/resource evidence; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and owner release approval. Synthetic/stub tests do not establish video accuracy or commercial readiness.
