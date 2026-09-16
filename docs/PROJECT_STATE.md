# Project state — 2026-09-16

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed Open Model Zoo person detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent.

`analytics_lab.video` owns bounded ordinary-local-file decode and timestamps. `analytics_lab.perception` owns validated boxes/keypoints, temporary session-local tracks and explicit posture abstention. `analytics_lab.openvino_omz` owns exact artifact verification plus the reviewed OpenVINO/OMZ detector-pose adapter. `analytics_lab.openvino_pipeline` composes perception with fresh tracking/temporal state. `analytics_lab.evaluation` measures one-to-one matches, misses, false alerts, false alerts per camera-hour and alert delay across site-scoped cameras. `analytics_lab.validation` binds rights references, media SHA-256 identities, model/runtime/device identities and throughput evidence without retaining video. `analytics_lab.validation_cli` exposes that suite through a strict local JSON manifest.

## Current acceptance-moving change
Multi-sample validation now prepares the reviewed OpenVINO backend once per suite instead of re-verifying and recompiling the same models for every clip. Each sample still receives a fresh perception/tracker/temporal adapter, so session-local tracks cannot bleed between clips. Preparation latency is recorded separately and remains included in total suite elapsed time and aggregate throughput.

A deterministic focused regression using two samples reduced runtime-factory/model-preparation calls from the prior per-sample behavior of 2 to 1 while proving two distinct per-sample adapters shared only the prepared backend. The injected clock values are software-test fixtures, not hardware performance evidence. No real model, runtime package, public media, customer media, or home-camera footage was downloaded or executed for this change.

## Reviewed donor/provenance baseline
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`.
- Pinned FP16 artifacts: `person-detection-retail-0013` and `human-pose-estimation-0001`, admitted only by exact size/SHA-384 through the local artifact manifest.
- OpenVINO Runtime release baseline: tag `2026.3.1`, commit `759c5a6ab8c066af5f4bc5ebd04643706012a37d`, Apache-2.0 source lineage. Platform wheel hashes, native dependency notices and final redistribution packaging remain release gates.
- RTMLib commit `03a1693e59e4f7cd84582c0fb30459b3bf18ad42` remains code-only; its default HumanArt-trained detector weights stay hold/do-not-ship because commercial/redistribution rights remain unresolved in reviewed evidence.

## Verified repository baseline
Main before the current work item: `5b9defa6bb1a20b1cc92d69c22b2a1f2f57685d1`. Its Linux regression, Windows regression and `Analytics quality gate` completed successfully. The preceding site/camera scoping regression is integrated: the same local camera ID may be used at different sites, while overlap within one site/camera still fails closed.

The current reuse change was syntax-compiled locally and a deterministic focused regression passed before repository mutation. Exact-head GitHub Linux, Windows and Analytics quality-gate checks remain authoritative before merge.

## Efficiency / execution ledger
One worker, one active acceptance-moving work item, at most one implementation PR. Current work item: amortize reviewed runtime/model initialization across a validation suite without sharing tracking state. Base: `5b9defa6bb1a20b1cc92d69c22b2a1f2f57685d1`. Open implementation PRs at intake: 0. Active runs for the work-item head before mutation: 0. Unchanged retries: 0. CI-triggering requests before opening the implementation PR: 0. Consecutive sessions without tested acceptance improvement: 0.

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

## Next executable step
Run one bounded CPU validation suite with externally provisioned OpenVINO `2026.3.1`, the exact pinned OMZ artifacts, and authorized labeled positive/negative local clips. Record exact media/model/runtime/device identities, preparation time, processing throughput, misses, false alerts per camera-hour, alert delay and per-site/per-camera results. If prone-person recall is inadequate, compare the current detector-isolated four-keypoint method with the pinned full OpenPose decoder before adding another framework.

## Outstanding commercial-release gates
Actual rights-reviewed runtime execution; platform wheel/native dependency provenance; authorized held-out positive and normal-negative video; false-alert and missed-event measurements; documented failure cases; comparable latency/resource evidence; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and owner release approval. Synthetic/stub tests do not establish video accuracy or commercial readiness.
