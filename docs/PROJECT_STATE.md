# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The person-down **candidate** path under measured validation is:

`authorized local video -> reviewed detector -> spatial continuity -> reviewed OpenPose model -> bounded pose association -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> labeled evaluation/aggregation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media hashes, reviewed model/runtime identities, decoded coverage, continuity, timing and evaluation metrics without retaining video in public GitHub.

## Current acceptance-moving work
Live `main` entering the current work item is `01ba93ce80d63bb2f512b75f7ccc2d9c752870db`; its post-merge Analytics quality run passed on attempt 1. There was no open implementation PR at intake. The single current evidence branch is `evidence/person-down-unknown-gap` and it attacks only the measured temporal persistence fragmentation.

### Accepted staged-real temporal evidence through PR #32
The first end-to-end run with historical temporal defaults (3000 ms down duration, 750 ms maximum inter-observation gap, 4 minimum samples and 0.70 minimum confidence) produced a legitimate staged-real miss: **1 labeled positive episode, 0 matched events, 1 miss and 0 false alerts** across the bounded two-clip seed. The positive clip produced 85 `down` posture frames, but all 85 were below the historical 0.70 temporal confidence floor.

The one measured confidence correction changed only temporal `min_confidence` to the already-reviewed posture keypoint floor of **0.10**. On the same exact media/model/runtime identities it converted the positive `down` postures into **85 qualifying down frames** while the hard negative remained **0 down / 0 candidate events / 0 false alerts**. The positive still missed because persistence was fragmented: the longest uninterrupted qualifying run was only **464 ms / 15 frames**, versus the unchanged 3000 ms requirement. Measured run breaks were dominated by **14 `unknown` posture interruptions**, with **1 `other` interruption** also observed. End-to-end CPU throughput remained about **10.1 FPS** on this tiny seed.

This evidence changes the next target: confidence is no longer the temporal blocker. The dominant measured error is short `unknown` fragmentation inside an otherwise recovered down-posture episode.

### One bounded unknown-gap correction under test
The temporal engine now has an explicit, default-off `max_unknown_gap_ms` setting. Default behavior remains fail-closed: `unknown` interrupts persistence when the value is zero. For this evidence-only measurement, the existing reviewed **750 ms max-gap budget** is reused as the maximum total span from the last accepted `down` observation to the next accepted `down` observation across one or more `unknown` observations.

The correction is intentionally narrow:

- `unknown` may bridge only when a down run is already active and the complete gap remains <= 750 ms;
- `upright` and `other` still reset immediately;
- low-confidence `down` still resets immediately;
- an unknown span that exceeds the bound resets before the next down sample;
- 3000 ms persistence, 4-sample minimum, 0.10 measured confidence floor, TTL and capacity remain unchanged;
- default production-facing behavior remains `max_unknown_gap_ms=0` until evidence justifies promotion.

Retain this correction only if the exact same staged-real positive produces a matched candidate without creating a hard-negative candidate/false alert. The diagnostic records bridged-unknown count, longest bridged gap, reset causes, candidate/match/miss/false-alert results, alert delay, continuity and CPU timing. If this still misses, the next run attacks the remaining measured reset source rather than lowering persistence blindly or model-shopping.

## Measured perception baseline
The reviewed `person-detection-0200` detector at confidence 0.10 plus evidence-only spatial continuity recovered the low-confidence prone-person trajectory on this deliberately tiny staged-real seed:

- Positive fall interval: **96/96 frames = 100% coverage**; **94/95 transitions = 98.95% linked**.
- Positive overall: **173/174 frames = 99.43% coverage**; **1 reset / 171 transitions = 0.58%**.
- Hard negative: **212/212 frames = 100% coverage**; **211/211 transitions = 100% linked**; **0 resets**.

Bounded pose association improved fall-window association from **36/96 to 90/96** and post-fall from **0/24 to 23/24**, with zero ambiguous fallback frames. The retained three-keypoint posture fallback materially reduced positive `unknown` output while preserving **0 down** on the hard negative. These are seed-specific engineering measurements, not general detector/tracker accuracy claims.

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

Current work item intake: open implementation PRs **0**, active runs for live main **0**, unchanged retries **0**, CI-triggering requests in this work session **0/2**, consecutive sessions without tested acceptance progress **0**. Branch preparation occurred before opening the single PR so the coherent change can trigger one exact-head CI/evidence run rather than serial invalidations.

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
Run the bounded 750 ms `unknown`-gap correction on the identical positive and hard-negative staged-real clips. Preserve exact candidate/match/miss/false-alert/delay, bridged-unknown, longest-gap, reset, continuity and CPU timing evidence. Keep the change only if it improves end-to-end acceptance without hard-negative regression. Do not lower the 3000 ms persistence requirement, bridge `upright`/`other`, retrain, or add another perception model unless this measurement proves the bounded correction inadequate.

Merge only when exact-head Linux, Windows and Analytics quality are green on the unchanged tested base and the bounded real-video evidence lane completes. Main post-merge verification intentionally does not reacquire real-video/model evidence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; false-alert and missed-event measurements at meaningful scale; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; platform/native dependency provenance; versioned installable integration adapter; packaging/notices; and explicit owner release approval. This two-clip staged-real seed does not establish commercial accuracy or commercial readiness.
