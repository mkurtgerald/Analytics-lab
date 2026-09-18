# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The person-down **candidate** path under measured validation is:

`authorized local video -> reviewed detector -> spatial continuity -> reviewed OpenPose model -> bounded pose association -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> labeled evaluation/aggregation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media hashes, reviewed model/runtime identities, decoded coverage, continuity, timing and evaluation metrics without retaining video in public GitHub.

## Accepted staged-real end-to-end milestone
PR #33 was accepted and merged to `main` at `5a63abd6201a7739d7c4cd8eb95a46fdbc77e67d`. Its unchanged exact head passed Linux, Windows, the Analytics quality gate and the bounded rights-cleared real-video lane on attempt 1. Post-merge Linux, Windows and Analytics quality also passed on the merged main revision; the real-video lane correctly did not reacquire media/models on main.

On the pinned Subject-1 two-clip GMDCSA-24 seed, the retained evidence-only temporal configuration produced the first matched staged-real person-down candidate:

- labeled positive episodes: **1**;
- candidate events: **1**;
- matched events: **1**;
- misses: **0**;
- hard-negative false alerts: **0**;
- decoded camera-hours: **0.0036**;
- median alert delay: **4024 ms** (one matched event, not a distribution);
- end-to-end CPU throughput: about **10.11 FPS**.

The positive sample contained **85 qualifying `down` frames**. The bounded temporal correction bridged **21 `unknown` frames**, with a longest bridged gap of **96 ms**, and recovered a **3008 ms / 70-sample** qualifying run while keeping `upright`, `other`, low-confidence `down`, expiry and over-bound gaps fail-closed. The raw uninterrupted `down` run remained only **464 ms / 15 frames**. The hard negative produced **0 `down`, 0 candidate events and 0 false alerts**.

This is a tiny staged-real engineering result, not a commercial accuracy claim. The largest remaining acceptance gap is now **evaluation breadth/integrity**, not another perception or temporal tweak on Subject 1.

## Current acceptance-moving work
Live `main` entering this work item is `5a63abd6201a7739d7c4cd8eb95a46fdbc77e67d`; its post-merge Linux, Windows and Analytics quality checks passed. There was no open implementation PR at intake. The single evidence branch is `evidence/person-down-heldout-s2`.

The next measurement rotates the same bounded two-clip PR-only lane to a **disjoint Subject-2 pair** rather than increasing CI duration or downloading a corpus:

- positive: `Subject 2/Fall/01.mp4`, 6 s, day, right-side fall from standing, annotated falling interval **1.4–6.0 s**, Git blob `6f308a3873d5768ea81b932d45a521e520368592`, exact size **6,472,725 bytes**;
- hard negative: `Subject 2/ADL/12.mp4`, 7 s, day, going to bed and **sleeping belly-down**, Git blob `aa83cc15559687e49c40afad920949be7befa61d`, exact size **7,791,870 bytes**.

The pair is intentionally under the existing 16 MiB aggregate media-selection bound and is similar in decoded duration to the accepted seed, preserving the five-minute hosted-Linux budget. The prone sleeping negative is chosen to stress person-down specificity rather than provide an easy upright-only negative. Both assets must pass exact Git-blob verification and local SHA-256 hashing before inference; SHA-256 identities are emitted by the private ephemeral evidence run and media is never uploaded to public GitHub.

Retain the current algorithm unchanged for this held-out run. If the positive misses or the prone normal-negative alerts, attack the largest measured error source on these held-out clips before changing thresholds or adding another model. If both behave acceptably, expand to another disjoint subject/setup while keeping file-level acquisition and the same bounded resource envelope.

## Measured perception baseline from accepted Subject-1 seed
The reviewed `person-detection-0200` detector at confidence 0.10 plus evidence-only spatial continuity recovered the low-confidence prone-person trajectory on the accepted seed:

- positive fall interval: **96/96 frames = 100% coverage**; **94/95 transitions = 98.95% linked**;
- positive overall: **173/174 frames = 99.43% coverage**; **1 reset / 171 transitions = 0.58%**;
- hard negative: **212/212 frames = 100% coverage**; **211/211 transitions = 100% linked**; **0 resets**.

Bounded pose association improved fall-window association from **36/96 to 90/96** and post-fall from **0/24 to 23/24**, with zero ambiguous fallback frames. The retained three-keypoint posture fallback materially reduced positive `unknown` output while preserving **0 down** on the hard negative. These remain seed-specific measurements.

## Evidence/data provenance
### GMDCSA-24
- Primary repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`, authored 2024-08-21 and aligned with the v2.1 data update.
- Repository license at that revision: MIT.
- Versioned dataset provenance: Zenodo v2.1 DOI `10.5281/zenodo.13354453`; associated paper DOI `10.1016/j.dib.2024.110892`.
- Accepted Subject-1 positive: `Subject 1/Fall/05.mp4`, Git blob `4e13ed24f6c2af7062b64b1920ef706c51efe94b`, local SHA-256 `1aad4e2ee93b5498283fdc4a0478a5913dffc2ac8e2538574cd821ffb1b485f1`, positive interval 1.8–5.0 s.
- Accepted Subject-1 hard negative: `Subject 1/ADL/15.mp4`, Git blob `903da9245132cf70c10124edd0625c958f702cb8`, local SHA-256 `2cf0d421cfd8e34280bf02afc67a4f1d1abee3cd87a23f11e69ef393fedc6fdb`.
- Active held-out Subject-2 assets are listed above and must be locally SHA-256 hashed before any result is admitted.
- Media stays outside public GitHub and is re-hashed before and after measured execution.

### Runtime/model provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Pose artifact: `human-pose-estimation-0001` FP16, exact-size/hash reviewed.
- Evidence detector: `person-detection-0200` FP16, exact-size/hash reviewed.
- OpenVINO Runtime: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- Reference decoder source: `demos/common/python/model_zoo/model_api/models/open_pose.py` at the same pinned OMZ commit; adapted diagnostic code preserves Intel copyright and Apache-2.0 attribution.

## Efficiency / execution ledger
One worker, one acceptance-moving work item, at most one implementation PR. No additional model family, training job, paid resource, self-hosted runner, home/customer media or duplicate agent is introduced.

Current work item intake: open implementation PRs **0**, active runs for live main **0**, unchanged retries **0**, CI-triggering requests in this work session **0/2**, consecutive sessions without tested acceptance progress **0**. The local preflight for the implementation action passed from fresh live counts. Branch preparation occurs before opening the single PR so the coherent held-out rotation triggers one exact-head CI/evidence run rather than serial invalidations.

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
Run the unchanged accepted algorithm on the disjoint Subject-2 positive and prone sleeping hard negative. Preserve exact media SHA-256 identities, positives, candidate/match/miss/false-alert counts, alert delay, detector/association/posture/temporal failure traces, decoded camera-hours, preparation/inference timing and CPU throughput. Do not tune against Subject 2 before recording the first held-out result.

Merge only when exact-head Linux, Windows and Analytics quality are green on the unchanged tested base and the bounded real-video evidence lane completes. Main post-merge verification intentionally does not reacquire real-video/model evidence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across genuinely different cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; false-alert and missed-event measurements at meaningful scale; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; platform/native dependency provenance; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Small GMDCSA-24 subject rotations do not establish commercial accuracy or commercial readiness.
