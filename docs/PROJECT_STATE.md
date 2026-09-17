# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed Open Model Zoo person detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed OMZ artifact identities, OpenVINO runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video. Media is re-hashed after sample execution so changed media cannot inherit stale evidence.

## Current acceptance-moving work
The bounded real-world validation seed is executable and a narrowly scoped hosted execution lane is being validated on `evidence/*` pull-request branches. The lane remains inside the existing Linux five-minute budget, read-only GitHub permissions and hosted Ubuntu runner. It installs only pinned OpenVINO `2026.3.1` plus pinned headless OpenCV, then invokes the repository's bounded seed preparer and validation CLI. Seed media/model bytes live only below the ephemeral runner temp directory and are never uploaded as workflow artifacts.

This execution lane exists solely to break the real-evidence gate. Main pushes and ordinary PRs do not run it. It does not change commercial-release authority, licensing boundaries, home/customer-media prohibitions, self-hosted-runner prohibitions or the requirement that real-video metrics come from exact rights-reviewed inputs.

## Reviewed data-source baseline
### UE4 Fall Detection Dataset — synthetic positive/negative development source
- Primary repository: `carolinehuang033/UE4_Fall_Detection_Dataset`.
- Pinned revision: `55041766dea68eaddc1df1c06aabd0a51931a22a`.
- Primary `LICENSE.txt` at that revision states CC BY 4.0 and expressly permits adaptation for any purpose, including commercially, with attribution.
- Media origin: **synthetic** (Unreal Engine 4 generated).
- Eligible engineering use: commercial training/evaluation subject to attribution and exact-asset identity.
- Limitation: synthetic observations and videos do not establish real-video accuracy.

### GMDCSA-24 — preferred bounded real-world validation source
- Primary data repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`.
- Repository `LICENSE` at that revision is MIT and requires preservation of its copyright/permission notice. The associated Data in Brief paper is open access under CC BY 4.0, identifies dataset DOI `10.5281/zenodo.13354453`, states that the dataset can be used to train or test a fall-detection system, and reports 81 fall plus 79 ADL clips from four subjects in three home setups.
- Media origin: **real_world**, staged by consenting actors.
- Initial bounded seed:
  - positive: `Subject 1/Fall/05.mp4`, repository blob `4e13ed24f6c2af7062b64b1920ef706c51efe94b`, 5,872,655 bytes, walking followed by a right-side fall, falling interval 1.8 s through 5 s;
  - hard negative: `Subject 1/ADL/15.mp4`, repository blob `903da9245132cf70c10124edd0625c958f702cb8`, 7,233,079 bytes, walking/picking an object from the ground/sitting.
- The seed preparer verifies Git-blob identity, computes local SHA-256, preserves attribution, and emits the strict validation manifest. Media is not committed to Analytics Lab.

### Figshare Video-Based Fall Detection Dataset with 2017 Activities from 29 Subjects — real-world secondary source
- Figshare article `28596332`, version 2, posted 2025-03-14.
- Creator: Ivan Ursul.
- Source page reports 2,017 recordings including 999 falls and 1,017 activities of daily living.
- Source page license: CC BY 4.0.
- Media origin: **real_world**.
- The complete dataset is about 2.36 GB and remains outside the current bounded-resource path until the small GMDCSA-24 seed proves the runtime path.

UR Fall remains excluded from the commercial path because its official source states non-commercial terms. Earlier Roboflow mirrors with unclear upstream image provenance remain hold-only.

## Reviewed perception/runtime baseline
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`.
- Pinned FP16 artifacts: `person-detection-retail-0013` and `human-pose-estimation-0001`, admitted only by exact size/SHA-384 through the local artifact manifest.
- OpenVINO Runtime baseline: `2026.3.1`, Apache-2.0 source lineage.
- Hosted evidence lane decoder: `opencv-python-headless==4.12.0.88`, Apache-2.0 distribution, used only for ephemeral validation execution.
- RTMLib remains code-only; its default HumanArt-trained weights remain hold/do-not-ship.

## Efficiency / execution ledger
One worker, one acceptance-moving work item, at most one implementation PR. Live base at this session intake: `8f56b9f4c0840915b20904c2557856e129e430ea`. Main Analytics quality run `35241293554` completed successfully on first attempt. Open implementation PRs at intake: 0. Active runs for the intake head: 0. Unchanged retries: 0. CI-triggering requests before opening this implementation PR: 0. Consecutive sessions without tested acceptance improvement: 0. Fresh verified pre-mutation facts permit implementation under the current policy.

Current implementation vehicle: `evidence/real-video-cpu`. The change adds only an evidence-branch conditional step to the existing hosted Linux job plus regression coverage and control documentation; it does not create another worker, workflow, runner or implementation lane.

## Reproduce
Dependency-free repository checks:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Prepare the bounded real-world evidence seed on an authorized internet-connected no-spend machine:

```sh
python -m analytics_lab.validation_seed --output-dir /path/to/private-validation
```

With OpenVINO Runtime `2026.3.1` provisioned in that environment, execute:

```sh
python -m analytics_lab.validation_cli --manifest /path/to/private-validation/validation-manifest.json
```

An `evidence/*` pull-request branch may perform those two commands on the hosted Linux runner after installing the pinned reviewed CPU runtime/decoder. The workflow emits aggregate JSON only and uploads no media/model artifact.

## Next executable step
Open the single evidence PR and let the exact-head hosted Linux job execute the bounded seed and reviewed CPU path. Inspect the first deterministic failure or, if successful, preserve the exact aggregate result: decoded duration, positives, misses, false alerts/camera-hour, alert delay, throughput, tracking continuity and concrete failure cases. Do not merge or claim performance until Linux, Windows and Analytics quality gate are green on the same exact head.

If the initial positive is missed, identify whether failure occurs at person detection, pose, track continuity, posture classification or temporal persistence before changing the stack. If prone-person detector recall is inadequate, quantify that detector-stage failure before comparing at most three rights-cleared alternatives. Do not retrain a commodity detector from scratch without measured need.

## Outstanding commercial-release gates
Actual reviewed runtime execution; expanded held-out positive/negative real-video evidence across cameras/sites; false-alert and missed-event measurements; documented failure cases; comparable latency/resource evidence; platform wheel/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and owner release approval. Synthetic/stub tests do not establish video accuracy or commercial readiness.
