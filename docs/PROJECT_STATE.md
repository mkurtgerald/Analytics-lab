# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed Open Model Zoo person detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed OMZ artifact identities, OpenVINO runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video. Media is re-hashed after sample execution so changed media cannot inherit stale evidence.

## Current acceptance-moving work
PR #24 / `evidence/detector-stage-attribution` is the single implementation vehicle. It follows the first successful real-video evidence run from merged PR #23 and measures the exact stage causing the missed positive before any donor/model change.

Exact head `087e114d8f6ed060190b9a2de78341c4abeeea1f` completed hosted run `35256762133` successfully on the first attempt. Guardrails passed, all 172 repository tests passed except the expected optional-OpenCV skip before runtime installation, the rights-bound real-video evidence step passed, Windows passed, and the final Analytics quality gate passed. The evidence step used OpenVINO `2026.3.1-22476-759c5a6ab8c-releases/2026/3`, OpenCV `4.12.0`, NumPy `2.2.6`, the exact two GMDCSA-24 seed clips, and the four reviewed OMZ artifacts.

The real validation result remains a miss rather than a commercial accuracy claim. Across 386 decoded frames / 12.96 seconds, the positive produced zero candidate events and one missed episode (recall 0.0); the hard ADL negative produced zero false alerts. Aggregate throughput on this run was 10.90 FPS including model work and 508.99 ms of one-time model preparation. No alert-delay value exists because no event fired.

The new detector-stage attribution identifies two earlier perception failures:

- **Detector recall collapses as the positive becomes prone/low.** `gmdcsa24-s1-fall-05` decoded 174 frames. Before the annotated 1.8-5.0 s fall interval, the detector produced 54 detections and detected a person in 53/53 frames. During the annotated interval it produced only 21 detections and detected a person in 21/96 frames (21.9%). After 5.0 s it detected a person in 0/25 frames. Overall positive detection coverage was 74/174 frames (42.5%). The hard ADL negative remained detected in 212/212 frames with 223 detections. This quantitatively establishes inadequate prone/fall person recall for the current detector on this seed.
- **The simplified pose/posture baseline is also currently non-functional on this seed.** Every pose candidate classified `unknown`: positive 75/75 and hard negative 223/223, with zero `upright`, `down`, or `other`. Therefore even positive frames that survive detection cannot satisfy temporal `down` persistence. This must be fixed after the detector boundary is improved.
- Temporary tracking is not the first blocker. The positive produced three distinct temporary tracks over 75 assignments and the negative four over 223 assignments; detector and pose/posture failures occur earlier in the mandated attack order.

The diagnostic reuses the strict rights-bound validation manifest, exact media SHA-256 checks before and after execution, the verified OMZ backend, bounded local-file decoder, existing detector parser, pose extraction, IoU tracker and posture classifier. It emits aggregate counts only; no frames, images, paths or media/model artifacts are uploaded.

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
- Exact admitted local SHA-256 identities are `1aad4e2ee93b5498283fdc4a0478a5913dffc2ac8e2538574cd821ffb1b485f1` for the positive and `2cf0d421cfd8e34280bf02afc67a4f1d1abee3cd87a23f11e69ef393fedc6fdb` for the hard negative.
- The source annotation tables report nominal clip lengths of 5 seconds and 7 seconds respectively. Those source-level integer lengths are useful provenance but are not exact decoded media coverage; the checksum-bound decoder clock is used for evaluation duration and camera-time accounting.
- The seed preparer verifies Git-blob identity, computes local SHA-256, preserves attribution, and emits the strict validation manifest. Media is not committed to Analytics Lab.
- Independent MP4s use independent validation-stream camera IDs because their timestamps are clip-local rather than synchronized windows from one continuous recorder. This prevents false overlap rejection and false camera-hour double counting without asserting that the physical source cameras differ.

### Figshare Video-Based Fall Detection Dataset with 2017 Activities from 29 Subjects — real-world secondary source
- Figshare article `28596332`, version 2, posted 2025-03-14.
- Creator: Ivan Ursul.
- Source page reports 2,017 recordings including 999 falls and 1,017 activities of daily living.
- Source page license: CC BY 4.0.
- Media origin: **real_world**.
- The complete dataset is about 2.36 GB and remains outside the current bounded-resource path until the small GMDCSA-24 seed proves a replacement perception path worth broader evaluation.

UR Fall remains excluded from the commercial path because its official source states non-commercial terms. Earlier Roboflow mirrors with unclear upstream image provenance remain hold-only.

## Reviewed perception/runtime baseline
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`.
- Pinned FP16 artifacts: `person-detection-retail-0013` and `human-pose-estimation-0001`, admitted only by exact size/SHA-384 through the local artifact manifest.
- OpenVINO Runtime baseline: `2026.3.1`, Apache-2.0 source lineage.
- Hosted evidence lane decoder: `opencv-python-headless==4.12.0.88`, Apache-2.0 distribution, used only for ephemeral validation execution.
- Hosted execution resolved the runtime to `2026.3.1-22476-759c5a6ab8c-releases/2026/3`, OpenCV `4.12.0`, and NumPy `2.2.6`.
- RTMLib remains code-only; its default HumanArt-trained weights remain hold/do-not-ship.

## Efficiency / execution ledger
One worker, one acceptance-moving work item, at most one implementation PR. Current live base is `a41145c1fa24c3c230d3036b021a6fa83958e5dc`; PR #24 / `evidence/detector-stage-attribution` is the only implementation vehicle.

This session began with zero open implementation PRs, zero queued/running runs, zero unchanged retries, zero CI-triggering requests and zero consecutive no-progress sessions because PR #23 had just produced legitimate real-video metrics. PR #24 exact head `087e114d8f6ed060190b9a2de78341c4abeeea1f` consumed request 1/2 and produced new measured failure attribution on the first attempt; it was not a retry. This PROJECT_STATE update is batched as the second and final CI-triggering mutation permitted in the session. No extra worker, branch, workflow job, runner, media source, model, framework or paid resource is added.

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
python -m analytics_lab.detector_diagnostics --manifest /path/to/private-validation/validation-manifest.json
```

An `evidence/*` pull-request branch may perform these commands on the hosted Linux runner after installing the pinned reviewed CPU runtime/decoder. The workflow emits aggregate JSON only and uploads no media/model artifact.

## Next executable step
Finish exact-head verification for PR #24 and merge only if Linux real-video evidence, Windows regression and the final Analytics quality gate remain green on the unchanged tested base.

After merge, follow the mandated failure order. The current detector has now quantitatively failed prone/fall recall, so compare **at most three** commercially rights-cleared detector alternatives against this exact held-out two-clip seed on a comparable CPU/runtime envelope. Select the smallest candidate that materially restores detection during the annotated fall interval without unacceptable degradation on the hard negative. Do not retrain a commodity detector yet.

Once detector recall materially improves, attack the already measured pose/posture defect: the present simplified heatmap-to-posture path classified every real candidate `unknown`. Quantify whether the issue is keypoint confidence, detector-isolated crop geometry, simplified heatmap extraction or posture geometry before changing tracking or temporal persistence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across cameras/sites; materially improved detector recall; functional pose/posture discrimination on real video; false-alert and missed-event measurements at meaningful scale; alert-latency distribution once events fire; comparable latency/resource evidence; platform wheel/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Synthetic/stub tests and this two-clip staged real seed do not establish commercial accuracy or commercial readiness.
