# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path remains:

`authorized local video -> reviewed detector + pose model -> temporary tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed model-artifact identities, runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video.

## Current acceptance-moving work
Live `main` entering this work item is `b1af22f7a126dee106a2a1c45efc80031a01989b`; its post-merge Analytics quality run `35287570905` passed on attempt 1. At intake there were zero open implementation PRs and zero active runs for that head.

The single work item is now **posture quality on the continuity-selected, bounded-associated OpenPose track** on the exact same rights-cleared two-clip staged-real seed. PR #30 materially recovered pose association without hard-negative regression: fall-window association improved from 36/96 to 90/96 and post-fall from 0/24 to 23/24, while before-fall and the hard negative remained fully associated. The remaining measured error is that 26 of the 90 associated fall-window poses and 11 of the 23 post-fall poses still classify `unknown`.

The current evidence-only diagnostic leaves detector continuity, pose decoding, bounded association and production perception unchanged. It records aggregate `unknown` classification reasons plus required-keypoint count and comparable posture geometry summaries (required-keypoint confidence, torso fraction, torso orientation and bbox aspect) by posture and time window. No media/model artifact, frame, identity or source secret is emitted.

### Measured detector baseline
The reviewed `person-detection-0200` detector at confidence 0.10 plus evidence-only spatial continuity recovered the low-confidence prone-person trajectory on this deliberately tiny staged-real seed:

- Positive fall interval: **96/96 frames = 100% coverage**; **94/95 transitions = 98.95% linked**.
- Positive overall: **173/174 frames = 99.43% coverage**; **1 reset / 171 transitions = 0.58%**.
- Hard negative: **212/212 frames = 100% coverage**; **211/211 transitions = 100% linked**; **0 resets**.

Threshold lowering alone was rejected because duplicate/noisy boxes rose sharply. The continuity selector is not a general detector promotion; the current seed is single-person and does not prove crowded-scene identity separation.

### Measured pose / association baseline
The original detector-crop/global-heatmap-peak path was inadequate. The pinned Open Model Zoo full-frame preserve-aspect + 3x3 NMS + PAF-grouping path materially improved the same real-video seed and is retained as the current diagnostic pose path.

PR #30 retained one bounded, fail-closed association correction derived from the first exact-head geometry measurement: preserve the baseline overlap association, otherwise accept exactly one decoded pose with normalized bbox edge gap <= 0.125 selected-box diagonals; multiple eligible poses fail closed. On the identical seed:

- Positive before-fall association: **53/53**.
- Positive fall association: **90/96 = 93.75%**.
- Positive post-fall association: **23/24 = 95.83%**.
- Hard negative association: **212/212**, with **0 `down` classifications**.
- Fallback accepted **77** positive frames with **0 ambiguous fallback frames**; seven decoded-pose frames remained outside the bound.
- Corrected fall-window posture counts: **50 down, 13 upright, 1 other, 26 unknown**.
- Corrected post-fall posture counts: **12 down, 11 unknown**.

Association is therefore no longer the dominant measured error on this seed. Posture evidence quality on the recovered poses is next. This bounded diagnostic remains single-person staged-real evidence, not a general detector/association promotion or commercial accuracy claim.

## Evidence/data baseline
### GMDCSA-24 staged-real seed
- Primary repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`.
- Repository license at that revision: MIT. Associated Data in Brief paper: CC BY 4.0, DOI `10.5281/zenodo.13354453`.
- Positive seed: `Subject 1/Fall/05.mp4`, Git blob `4e13ed24f6c2af7062b64b1920ef706c51efe94b`, local SHA-256 `1aad4e2ee93b5498283fdc4a0478a5913dffc2ac8e2538574cd821ffb1b485f1`, annotated fall 1.8–5.0 s.
- Hard negative: `Subject 1/ADL/15.mp4`, Git blob `903da9245132cf70c10124edd0625c958f702cb8`, local SHA-256 `2cf0d421cfd8e34280bf02afc67a4f1d1abee3cd87a23f11e69ef393fedc6fdb`.
- Media remains outside public GitHub and is re-hashed around measured execution.

### Runtime/model provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Pose artifact: `human-pose-estimation-0001` FP16, already exact-size/hash reviewed.
- Evidence detector: `person-detection-0200` FP16, already exact-size/hash reviewed.
- OpenVINO Runtime: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- Reference decoder source: `demos/common/python/model_zoo/model_api/models/open_pose.py` at the same pinned OMZ commit; adapted diagnostic code preserves Intel copyright and Apache-2.0 attribution.

## Efficiency / execution ledger
One worker, one acceptance-moving work item, at most one implementation PR. Intake snapshot for this work item: `main=b1af22f7a126dee106a2a1c45efc80031a01989b`, open implementation PRs `0`, active runs for live main head `0`, unchanged retries `0`, CI-triggering requests this session `0`, consecutive sessions without tested acceptance progress `0`. The proposed evidence PR's automatic exact-head run is request **1/2**. No paid resource, self-hosted runner, home/customer media, new model family or additional worker is introduced.

## Reproduce
Repository checks:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Rights-bound evidence path in an authorized internet-connected no-spend environment:

```sh
python -m analytics_lab.validation_seed --output-dir /path/to/private-validation
python -m analytics_lab.validation_cli --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.detector_diagnostics --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.detector_thresholds \
  --manifest /path/to/private-validation/validation-manifest.json \
  --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.pose_diagnostics \
  --manifest /path/to/private-validation/validation-manifest.json \
  --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.openpose_diagnostics_v2 \
  --manifest /path/to/private-validation/validation-manifest.json \
  --candidate-dir /path/to/private-detector-cache
```

## Next executable decision
Run the exact-head posture-quality diagnostic on the same positive and hard-negative clips. Quantify why the corrected associated poses remain `unknown`: missing required shoulder/hip keypoints, required-keypoint confidence below the reviewed floor, or insufficient torso geometry. Compare the same geometry distributions with `down` and `upright` frames before changing any posture rule.

If one failure mode dominates and the distributions isolate a safe bounded correction, make one coherent diagnostic correction on this same PR, add a regression, and compare against the identical held-out seed. Keep it only if fall/post-fall classification materially improves without introducing hard-negative `down` regression. Do not lower thresholds blindly. If no safe separation exists, change approach rather than broadening the classifier.

Only after posture quality materially improves should this measured path be reconnected through temporal event validation. Do not start another detector/pose search, training or tracker rewrite while posture quality remains the measured largest error source.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; functional pose/posture discrimination on real video; safe pose-to-track association; false-alert and missed-event measurements at meaningful scale; alert-latency distribution once events fire; latency/resource envelope; platform/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Synthetic tests and this two-clip staged-real seed do not establish commercial accuracy or commercial readiness.
