# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed Open Model Zoo person detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent.

`analytics_lab.video` owns bounded ordinary-local-file decode and timestamps. `analytics_lab.perception` owns validated boxes/keypoints, temporary session-local tracks and explicit posture abstention. `analytics_lab.openvino_omz` owns exact artifact verification plus the reviewed OpenVINO/OMZ detector-pose adapter. `analytics_lab.openvino_pipeline` composes perception with fresh tracking/temporal state. `analytics_lab.evaluation` measures one-to-one matches, misses, false alerts, false alerts per camera-hour and alert delay across site-scoped cameras. `analytics_lab.validation` binds rights references, media SHA-256 identities, model/runtime/device identities and throughput evidence without retaining video. `analytics_lab.validation_cli` exposes that suite through a strict local JSON manifest.

## Current acceptance-moving change
Rights-bound validation now re-hashes each authorized local media file immediately after its sample execution and rejects the sample if the file disappeared, became a symlink, or no longer matches the declared SHA-256. Before this change, media was hashed only during suite preflight. A local file changed after preflight but before or during decode could therefore produce metrics that were still attributed to the original media identity.

The post-run integrity verification occurs after the measured inference clock stops, so checksum I/O does not distort the recorded sample inference elapsed time or frames-per-second evidence. A focused deterministic harness reproduced the mutation window and verified that changed media is rejected while unchanged media remains accepted. Repository regression coverage exercises both outcomes.

The previously merged validation protections remain in place: local decode uses the media/container timestamp clock when available; temporary IoU association uses maximum-cardinality matching; validation-manifest paths reject network/device namespaces; manifest-relative paths resolve against the manifest directory; evaluation matching maximizes one-to-one cardinality; validation duration is bound to decoded timestamp coverage; multi-sample validation prepares the reviewed OpenVINO backend once per suite while each sample gets fresh perception/tracker/temporal state.

## Positive-data rights review — bounded candidate pass
Three candidate positive-data sources were inspected and the search is stopped at the configured limit:

1. **Roboflow Universe Fall&Slips (SASCans)** — Universe currently labels the project MIT and exposes fallen-person / standing-person classes, but the project page publishes no upstream/source provenance. Hold until exact dataset-version provenance and the uploader's right to license the underlying imagery are independently established.
2. **UR Fall Detection Dataset** — a Roboflow mirror labels a derivative CC BY 4.0, but the official University of Rzeszow source states CC BY-NC-SA 4.0, non-commercial academic use, and requires separate contact for commercial use. Reject from the commercial validation/training path without separate permission and rights review.
3. **Fallen Person Dataset (FPDS), University of Alcala** — the originating paper describes the authors' public labeled dataset and the paper itself is CC BY 4.0, but the accessible primary evidence reviewed here does not state an explicit dataset license granting commercial reuse of the image corpus. Hold pending explicit dataset terms or owner-authorized rights review.

No candidate media was downloaded, retained, trained on, or added to GitHub. This pass does not authorize dataset use and is not a legal opinion.

## Reviewed donor/provenance baseline
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`.
- Pinned FP16 artifacts: `person-detection-retail-0013` and `human-pose-estimation-0001`, admitted only by exact size/SHA-384 through the local artifact manifest.
- OpenVINO Runtime release baseline: tag `2026.3.1`, commit `759c5a6ab8c066af5f4bc5ebd04643706012a37d`, Apache-2.0 source lineage. Platform wheel hashes, native dependency notices and final redistribution packaging remain release gates.
- RTMLib commit `03a1693e59e4f7cd84582c0fb30459b3bf18ad42` remains code-only; its default HumanArt-trained detector weights stay hold/do-not-ship because commercial/redistribution rights remain unresolved in reviewed evidence.

## Verified repository baseline
Main at intake: `55b38e2a007a300fc4bfa8d23e3225682fd1e043`. Its post-merge Analytics quality workflow completed successfully. No implementation PR was open at intake and no repository workflow run was queued or running.

Before mutation, `tools/guardrails.py preflight` was run with fresh verified counts and allowed implementation. A focused local media-identity harness passed 2/2 outcomes: changed media rejected; unchanged media accepted. Exact-head GitHub Linux, Windows and Analytics quality-gate checks remain authoritative before merge.

## Efficiency / execution ledger
One worker, one active acceptance-moving work item, at most one implementation PR. Current work item: bind executed validation media to the declared SHA-256 after sample execution. Base: `55b38e2a007a300fc4bfa8d23e3225682fd1e043`. Open implementation PRs at intake: 0. Active runs for the base/head before mutation: 0. Unchanged retries: 0. CI-triggering requests before opening the implementation PR: 0. Consecutive sessions without tested acceptance improvement: 0.

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

Relative `artifact_root` and `video_path` entries in that manifest are interpreted relative to the manifest file itself. Network/UNC/device namespace paths are rejected before validation media or model paths are touched. Local decode uses the media/container timestamp clock when available, with a bounded nominal-FPS fallback only when the media clock cannot advance monotonically. Validation media must match the declared SHA-256 at preflight and still match immediately after its inference run or the evidence is rejected.

## Next executable step
Provision the already-reviewed OpenVINO `2026.3.1` runtime and exact pinned OMZ artifacts in an authorized no-spend execution environment, then run bounded real CPU validation only after an explicit commercially usable positive-media source and authorized normal-negative source are established. Record exact media/model/runtime/device identities, preparation time, decoded evaluation duration, processing throughput, misses, false alerts per camera-hour, alert delay and per-site/per-camera results. If prone-person recall is inadequate, quantify the detector-stage failure before comparing at most three rights-cleared alternatives.

## Outstanding commercial-release gates
Actual rights-reviewed runtime execution; platform wheel/native dependency provenance; explicitly commercially usable held-out positive and normal-negative video; false-alert and missed-event measurements; documented failure cases; comparable latency/resource evidence; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and owner release approval. Synthetic/stub tests do not establish video accuracy or commercial readiness.
