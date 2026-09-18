# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The person-down **candidate** path under measured validation is:

`authorized local video -> reviewed detector -> spatial continuity -> reviewed OpenPose model -> bounded pose association -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> labeled evaluation/aggregation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media hashes, reviewed model/runtime identities, decoded coverage, continuity, timing and evaluation metrics without retaining video in public GitHub.

## Current acceptance-moving work
Live `main` entering PR #32 is `08720e96d0e49efcfcd0f94e9b27fe35af25dc87`; its post-merge Analytics quality run passed on attempt 1. PR #32 is the single implementation/evidence vehicle for first end-to-end staged-real temporal evidence.

### First exact-head end-to-end measurement
Exact head `5202180b35e28fbb33008906535f62d90be0a5ae` passed Linux, Windows, the Analytics quality gate and the bounded rights-cleared real-video lane on attempt 1. It used the existing temporal defaults unchanged: 3000 ms down duration, 750 ms maximum gap, 4 minimum samples and 0.70 minimum confidence.

The result is legitimate staged-real performance evidence, and it is a **miss**:

- Aggregate decoded evidence: **1 positive episode, 0 matched events, 1 miss, 0 false alerts, 0 candidate events, recall 0.0, 0 false alerts per decoded camera-hour** across **0.0036 decoded camera-hours**.
- Positive clip: **174 frames / 5824 ms**, 166 associated poses, 85 `down`, 66 `upright`, 22 `unknown`, 1 `other`; **all 85 down frames were below the 0.70 temporal confidence floor**.
- Positive down-confidence distribution: **min ~0.102, p50 ~0.161, p90 ~0.452, max ~0.561**. Therefore **0 qualified down frames** and no temporal run could begin.
- Hard negative: **212 frames / 7136 ms**, 212 associated poses, **0 down**, 181 upright, 30 other, 1 unknown, **0 candidate events / 0 false alerts**.
- End-to-end measured throughput on the two clips: **~10.10 FPS**, **38.23 s** measured sample execution, plus **~0.61 s** preparation.
- Tracking continuity remained strong on this seed: positive 173/174 selected with 170 linked transitions and 1 reset; hard negative 212/212 selected with 211 linked transitions and 0 resets.

This isolates the next temporal error source cleanly: the historical 0.70 temporal confidence gate is incompatible with the confidence semantics of the already-accepted posture path. It is not a calibrated probability; accepted real-video `down` posture confidence is bounded by the required OpenPose keypoint scores.

### One bounded temporal correction under test
The next exact head changes **only** temporal `min_confidence`, from 0.70 to the already-reviewed posture required-keypoint floor of **0.10**. Down-duration, max-gap, minimum-sample, TTL and capacity limits remain unchanged. The diagnostic also records the longest raw and qualifying down run so that, if persistence still blocks the event, the next decision is measured rather than guessed.

This correction is evidence-only and not production-promoted. Retain it only if it converts the previously rejected positive down frames into qualifying temporal evidence without creating a hard-negative candidate/false alert. If the positive event still misses, do not lower persistence blindly; use the exact longest qualifying run and reset causes to identify the next bounded temporal defect.

## Measured perception baseline
The reviewed `person-detection-0200` detector at confidence 0.10 plus evidence-only spatial continuity recovered the low-confidence prone-person trajectory on this deliberately tiny staged-real seed:

- Positive fall interval: **96/96 frames = 100% coverage**; **94/95 transitions = 98.95% linked**.
- Positive overall: **173/174 frames = 99.43% coverage**; **1 reset / 171 transitions = 0.58%**.
- Hard negative: **212/212 frames = 100% coverage**; **211/211 transitions = 100% linked**; **0 resets**.

Bounded pose association previously improved fall-window association from **36/96 to 90/96** and post-fall from **0/24 to 23/24**, with zero ambiguous fallback frames. The retained three-keypoint posture fallback produced the measured end-to-end positive total of 85 `down` frames while preserving **0 down** on the hard negative. These are seed-specific engineering measurements, not general detector/tracker accuracy claims.

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
One worker, one acceptance-moving work item, one implementation PR. No additional model family, training job, paid resource, self-hosted runner, home/customer media or duplicate agent is introduced.

PR #32 first exact-head measurement was **CI request 1/2** and passed on attempt 1; unchanged retries remain **0**. The measured confidence-floor correction is the single coherent **CI request 2/2** for this session. No further feature-branch CI-triggering mutation is permitted in this session. If the second exact-head run exposes another deterministic temporal blocker, preserve that evidence for the next session rather than pushing again.

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
Measure the 0.10 temporal confidence correction on the identical positive and hard-negative clips. Keep every other temporal threshold fixed. Preserve exact candidate/match/miss/false-alert/delay, longest-down-run, reset, continuity and CPU timing evidence. If the event still misses, the next run must attack the measured persistence/reset blocker rather than model-shopping, training or broad threshold tuning.

Merge only when exact-head Linux, Windows and Analytics quality are green on the unchanged tested base and the bounded real-video evidence lane completes. Main post-merge verification intentionally does not reacquire real-video/model evidence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; false-alert and missed-event measurements at meaningful scale; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; platform/native dependency provenance; versioned installable integration adapter; packaging/notices; and explicit owner release approval. This two-clip staged-real seed does not establish commercial accuracy or commercial readiness.
