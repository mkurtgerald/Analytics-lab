# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path remains:

`authorized local video -> reviewed detector + pose model -> temporary tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed model-artifact identities, runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video.

## Current acceptance-moving work
Live `main` entering this work item is `b1af22f7a126dee106a2a1c45efc80031a01989b`; its post-merge Analytics quality run `35287570905` passed on attempt 1. PR #31 is the single implementation/evidence vehicle. Its first exact head `56ab42ec45b17b7d56ca437d09d57621e1bfcc44` passed Linux, Windows, the Analytics quality gate and the bounded rights-cleared real-video lane on attempt 1.

The single work item is **posture quality on the continuity-selected, bounded-associated OpenPose track** on the exact same rights-cleared two-clip staged-real seed. PR #30 materially recovered pose association without hard-negative regression: fall-window association improved from 36/96 to 90/96 and post-fall from 0/24 to 23/24, while before-fall and the hard negative remained fully associated.

### First exact-head posture measurement
The remaining `unknown` error is now isolated to **missing decoded required shoulder/hip keypoints**, not the confidence, torso-size, orientation or aspect thresholds:

- Positive before-fall: **53/53 associated; 53 upright; 0 unknown**.
- Positive fall window: **90/96 associated; 50 down, 13 upright, 1 other, 26 unknown**. All 26 unknowns are `required_keypoints_missing`; required-point counts are **3 points: 19, 2 points: 5, 1 point: 2**.
- Positive post-fall: **23/24 associated; 12 down, 11 unknown**. All 11 unknowns are `required_keypoints_missing`; required-point counts are **3 points: 5, 2 points: 2, 1 point: 4**.
- Hard negative: **212/212 associated; 180 upright, 30 other, 2 unknown, 0 down**. Both unknowns have exactly three required points.
- Association fallback remains unchanged: 77 positive frames recovered, zero ambiguous fallback frames, and no hard-negative fallback use.

The 50 fall-window `down` poses have all four required points and strongly horizontal measured geometry (horizontal-fraction p50 ~0.993; decoded-pose width/height p50 ~2.43). The 12 post-fall `down` poses are similarly horizontal (p50 ~0.997; width/height p50 ~2.78). The unknown poses cannot reach the existing torso-geometry classifier because one or more required keypoints are absent.

### Bounded correction under test
The evidence lane now tests one fail-closed correction only for an existing `unknown` pose with **exactly three of the four required shoulder/hip keypoints**. With exactly one joint absent, one body side is necessarily complete. The diagnostic uses that complete same-side shoulder-to-hip vector with the **existing** required-keypoint floor (0.10), torso-fraction floor (0.10), orientation threshold (0.70), and bbox aspect threshold (1.15). It accepts only decisive `upright` or `down`; incomplete, low-confidence or ambiguous geometry remains `unknown`.

This correction does not alter production perception, detector continuity, OpenPose decoding or pose association. It is retained only if the identical real-video seed recovers a material share of the **24 positive three-point unknown frames** while the hard negative remains at **0 down**, before-fall remains unchanged, and no existing associated classification regresses. A useful acceptance bar for this bounded experiment is at least half of those 24 positive three-point frames becoming decisive with zero hard-negative down regression.

## Measured detector baseline
The reviewed `person-detection-0200` detector at confidence 0.10 plus evidence-only spatial continuity recovered the low-confidence prone-person trajectory on this deliberately tiny staged-real seed:

- Positive fall interval: **96/96 frames = 100% coverage**; **94/95 transitions = 98.95% linked**.
- Positive overall: **173/174 frames = 99.43% coverage**; **1 reset / 171 transitions = 0.58%**.
- Hard negative: **212/212 frames = 100% coverage**; **211/211 transitions = 100% linked**; **0 resets**.

Threshold lowering alone was rejected because duplicate/noisy boxes rose sharply. The continuity selector is not a general detector promotion; the current seed is single-person and does not prove crowded-scene identity separation.

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
- Pose artifact: `human-pose-estimation-0001` FP16, exact-size/hash reviewed.
- Evidence detector: `person-detection-0200` FP16, exact-size/hash reviewed.
- OpenVINO Runtime: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- Reference decoder source: `demos/common/python/model_zoo/model_api/models/open_pose.py` at the same pinned OMZ commit; adapted diagnostic code preserves Intel copyright and Apache-2.0 attribution.

## Efficiency / execution ledger
One worker, one acceptance-moving work item, one implementation PR. Intake: `main=b1af22f7a126dee106a2a1c45efc80031a01989b`, open implementation PRs `0`, active runs for live main `0`, unchanged retries `0`, CI-triggering requests `0`, consecutive sessions without tested acceptance progress `0`.

PR #31 exact-head measurement was CI request **1/2** and passed on attempt 1. The coherent three-point fallback correction is request **2/2** for this session. No unchanged retry, paid resource, self-hosted runner, home/customer media, new model family or additional worker is introduced. After the second exact-head run, no further feature-branch CI mutation is permitted in this session.

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
Run the exact-head three-point fallback correction on the same positive and hard-negative clips. Retain it only if it materially reduces positive fall/post-fall unknowns without creating a hard-negative `down` classification or changing already-decisive four-point classifications. If it fails that bar, stop this fallback rather than broadening it to two-point/bbox-only inference.

If retained and exact-head Linux, Windows and Analytics quality remain green, merge this evidence improvement. The next session should then reconnect the measured detector + association + posture path through temporal event validation so the first legitimate end-to-end staged-real person-down event metrics can be produced. Do not start another detector/pose search, training or tracker rewrite while this measured path remains viable.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; functional pose/posture discrimination on real video; safe pose-to-track association; false-alert and missed-event measurements at meaningful scale; alert-latency distribution once events fire; latency/resource envelope; platform/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Synthetic tests and this two-clip staged-real seed do not establish commercial accuracy or commercial readiness.
