# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The person-down **candidate** path under measured validation is:

`authorized local video -> reviewed detector -> spatial continuity -> reviewed OpenPose model -> bounded pose association -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> labeled evaluation/aggregation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media hashes, reviewed model/runtime identities, decoded coverage, continuity, timing and evaluation metrics without retaining video in public GitHub.

## Current acceptance-moving work
Live `main` entering this work item is `08720e96d0e49efcfcd0f94e9b27fe35af25dc87`. Its post-merge Analytics quality run `35290818608` completed successfully on attempt 1. Intake had **0 open implementation PRs**, **0 active runs on the live main head**, **0 unchanged retries**, **0 CI-triggering requests in this session**, and **0 consecutive sessions without tested acceptance progress**.

PR #31 already retained the evidence-only three-keypoint posture fallback after exact staged-real measurement. On the identical two-clip seed:

- Positive before-fall: **53 upright, 0 unknown**.
- Positive fall window: **70 down, 6 unknown** after the bounded posture correction.
- Positive post-fall: **14 down, 10 unknown** after the bounded posture correction.
- Hard negative: **0 down classifications**.
- Detector continuity remained the measured low-confidence `person-detection-0200` path; bounded pose association remained unchanged.

The single work item is now **first end-to-end staged-real temporal person-down evidence**. Branch `evidence/person-down-temporal-e2e` reconnects the existing measured detector + OpenPose + association + posture path to the existing `PersonDownEngine` and deterministic evaluator. The first measurement deliberately uses the existing default temporal configuration without tuning: **3000 ms down duration, 750 ms maximum gap, 4 minimum samples, 0.70 minimum confidence**.

The end-to-end diagnostic records, per exact clip and in aggregate: decoded duration/camera-hours, frames, association coverage, continuity/link/reset counts, posture/basis counts, down-confidence distribution, qualified versus low-confidence down frames, longest qualifying down run, temporal reset causes, candidate events, positives, matched events, misses, false alerts, exact alert-delay samples/median, detector/pose inference time, preparation/elapsed time and throughput/FPS. Missing/unknown posture frames are fed fail-closed into temporal logic rather than silently skipped.

This diagnostic is evidence-only. It does not promote the detector, pose model, association rule, posture fallback or temporal thresholds to commercial accuracy status. If the first exact-head run misses the positive event, the next correction must target the measured dominant temporal failure (for example confidence floor or interrupted persistence) rather than broadening perception or changing several thresholds at once.

## Measured perception baseline
The reviewed `person-detection-0200` detector at confidence 0.10 plus evidence-only spatial continuity recovered the low-confidence prone-person trajectory on this deliberately tiny staged-real seed:

- Positive fall interval: **96/96 frames = 100% coverage**; **94/95 transitions = 98.95% linked**.
- Positive overall: **173/174 frames = 99.43% coverage**; **1 reset / 171 transitions = 0.58%**.
- Hard negative: **212/212 frames = 100% coverage**; **211/211 transitions = 100% linked**; **0 resets**.

Threshold lowering alone was rejected because duplicate/noisy boxes rose sharply. Bounded pose association previously improved fall-window association from **36/96 to 90/96** and post-fall from **0/24 to 23/24**, with zero ambiguous fallback frames. The three-keypoint posture fallback then reduced positive `unknown` classifications while preserving zero hard-negative `down` classifications. These are seed-specific engineering measurements, not general detector/tracker accuracy claims.

## Evidence/data baseline
### GMDCSA-24 staged-real seed
- Primary repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`.
- Repository license at that revision: MIT. Associated Data in Brief paper: CC BY 4.0, DOI `10.5281/zenodo.13354453`.
- Positive seed: `Subject 1/Fall/05.mp4`, Git blob `4e13ed24f6c2af7062b64b1920ef706c51efe94b`, local SHA-256 `1aad4e2ee93b5498283fdc4a0478a5913dffc2ac8e2538574cd821ffb1b485f1`, annotated positive interval 1.8–5.0 s.
- Hard negative: `Subject 1/ADL/15.mp4`, Git blob `903da9245132cf70c10124edd0625c958f702cb8`, local SHA-256 `2cf0d421cfd8e34280bf02afc67a4f1d1abee3cd87a23f11e69ef393fedc6fdb`.
- Media stays outside public GitHub and is re-hashed before and after measured execution.

### Runtime/model provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Pose artifact: `human-pose-estimation-0001` FP16, exact-size/hash reviewed.
- Evidence detector: `person-detection-0200` FP16, exact-size/hash reviewed.
- OpenVINO Runtime: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- Reference decoder source: `demos/common/python/model_zoo/model_api/models/open_pose.py` at the same pinned OMZ commit; adapted diagnostic code preserves Intel copyright and Apache-2.0 attribution.

## Efficiency / execution ledger
One worker, one acceptance-moving work item, at most one implementation PR. No additional model family, training job, paid resource, self-hosted runner, home/customer media or duplicate agent is introduced.

For this temporal work item, the pre-mutation live-state snapshot was verified from `main=08720e96d0e49efcfcd0f94e9b27fe35af25dc87`: open implementation PRs `0`, active runs for live head `0`, unchanged retries `0`, CI-triggering requests `0`, sessions without progress `0`. The guardrail preflight allowed implementation. Opening the single evidence PR will be **CI request 1/2**. One further CI-triggering correction remains available only if the first measured run exposes a deterministic, acceptance-moving defect. An unchanged retry remains permitted only for a diagnosed transient infrastructure failure.

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
python -m analytics_lab.person_down_e2e_diagnostics \
  --manifest /path/to/private-validation/validation-manifest.json \
  --candidate-dir /path/to/private-detector-cache
```

## Next executable decision
Run the exact-head end-to-end diagnostic on the same positive and hard-negative clips with the existing default temporal thresholds. Preserve the raw first-attempt evidence even if it misses. If the positive event is matched and the hard negative remains clean, retain the measurement path and move next to broader held-out real-video evidence rather than tuning this seed. If the positive event is missed, attack only the largest measured temporal error source and compare the correction against the exact same evidence before retaining it. Do not start another detector/pose search, tracker rewrite or training path while this measured path remains viable.

Merge only when exact-head Linux, Windows and Analytics quality are green on the unchanged tested base and the bounded real-video evidence lane completes. Main post-merge verification intentionally does not reacquire real-video/model evidence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; false-alert and missed-event measurements at meaningful scale; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; platform/native dependency provenance; versioned installable integration adapter; packaging/notices; and explicit owner release approval. This two-clip staged-real seed does not establish commercial accuracy or commercial readiness.
