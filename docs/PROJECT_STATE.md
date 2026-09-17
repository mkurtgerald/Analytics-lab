# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed Open Model Zoo person detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed OMZ artifact identities, OpenVINO runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video. Media is re-hashed after sample execution so changed media cannot inherit stale evidence.

## Current acceptance-moving work
PR #23 is the single implementation vehicle for the bounded hosted real-video CPU evidence lane. Its first exact-head run proved the previously blocked execution prerequisites: the hosted Ubuntu runner installed pinned OpenVINO `2026.3.1` and headless OpenCV, the bounded GMDCSA-24 clips and four reviewed OMZ artifacts were acquired and verified, and the strict validation manifest was emitted. The run then failed deterministically before inference because both independent MP4 files were represented as overlapping 0-based intervals on one camera timeline.

That failure is now root-caused rather than retried. The seed manifest represents each independent MP4 as its own validation stream while preserving the common GMDCSA-24 site/source identity. This keeps clip-local timestamps honest and prevents aggregate camera-hour accounting from double-counting an invented shared timeline. No acceptance threshold, rights check, checksum check or runtime gate is weakened.

The evidence lane remains inside the existing Linux five-minute budget, read-only GitHub permissions and hosted Ubuntu runner. Seed media/model bytes live only below the ephemeral runner temp directory and are never uploaded as workflow artifacts. Main pushes and ordinary PRs do not run the real-video step.

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
- Independent MP4s use independent validation-stream camera IDs because their timestamps are clip-local rather than synchronized windows from one continuous recorder. This prevents false overlap rejection and false camera-hour double counting without asserting that the physical source cameras differ.

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
- First hosted run resolved the runtime to `2026.3.1-22476-759c5a6ab8c-releases/2026/3`, OpenCV `4.12.0`, and NumPy `2.2.6` before the manifest overlap defect stopped validation.
- RTMLib remains code-only; its default HumanArt-trained weights remain hold/do-not-ship.

## Efficiency / execution ledger
One worker, one acceptance-moving work item, at most one implementation PR. Live base at this session intake: `8f56b9f4c0840915b20904c2557856e129e430ea`. Main Analytics quality run `35241293554` completed successfully on first attempt. Open implementation PRs at intake: 0. Active runs for the intake head: 0. Unchanged retries: 0. Consecutive sessions without tested acceptance improvement: 0.

Current implementation vehicle: PR #23 / `evidence/real-video-cpu`. First PR-head run `35245175574` was a deterministic first-attempt failure in the new evidence step after all ordinary guardrail/synthetic tests passed. No unchanged retry was used. Root cause: the two independent clip-local intervals were assigned one camera ID, triggering the existing fail-closed overlap guard before inference. The correction is a changed-head regression fix, not a retry. The next PR-head run is the second and final CI-triggering request allowed for this work item in the current session; if another deterministic failure appears, diagnose it without a third push this session.

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
Allow the changed exact PR head to execute once. If successful, preserve the exact aggregate real-video result: decoded duration, positives, misses, false alerts/camera-hour, alert delay, throughput, tracking continuity and concrete failure cases, then require Windows and Analytics quality gate green on the same head before merge. If another deterministic failure appears, stop pushing for this session, identify the precise stage from the logs/source path and carry the diagnosed correction into the next bounded session.

If the initial positive is missed, identify whether failure occurs at person detection, pose, track continuity, posture classification or temporal persistence before changing the stack. If prone-person detector recall is inadequate, quantify that detector-stage failure before comparing at most three rights-cleared alternatives. Do not retrain a commodity detector from scratch without measured need.

## Outstanding commercial-release gates
Actual reviewed runtime execution; expanded held-out positive/negative real-video evidence across cameras/sites; false-alert and missed-event measurements; documented failure cases; comparable latency/resource evidence; platform wheel/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and owner release approval. Synthetic/stub tests do not establish video accuracy or commercial readiness.
