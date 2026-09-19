# Project state — 2026-09-19

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

### Product priority
1. Detection + tracking — North Star foundation.
2. LPR/OCR.
3. Face — including selectable face blurring as a required privacy capability: policy-driven default blur, authorized unblur/reblur, server-side enforcement, permission gating, auditable state changes, and preservation of original evidence under retention/chain-of-custody rules.
4. Weapons.
5. Appearance search.

Person-down and slip/fall remain required deliverables. They are secondary only in sequencing; preserve their working path, regressions and evidence, and continue them when shared perception/tracking/evaluation work advances them without displacing the higher-priority acceptance item.

## Retained person-down path — secondary validation analytic
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media/model/runtime identities and aggregate measurements while source media remains ephemeral and outside public GitHub.

### Accepted person-down evidence frontier
The retained detector/temporal path is stable; repeated misses localize primarily to pose representation/association/posture continuity. This remains useful shared-perception evidence but does not own the critical path.

- GMDCSA Subject 1 produced a legitimate untouched positive emission with hard negatives clean; alert delay was 3992 ms, detector coverage 150/150, safe pose associations 175/181 overall, 107 qualified down observations and a 4096 ms longest qualified run. This is engineering evidence, not commercial accuracy.
- Subjects 2-4 remain protected evaluation holdouts. Subject 2's later night/cross-view positive had 76/82 detector selections but only 32/82 safe pose associations and a 736 ms longest qualified run; its negative had 226/226 associations and 0 false alerts. Subject 3 remained 0 candidates / 1 scored miss while its sitting-to-sleeping negative stayed clean. Subject 4 had 129/129 detector selection/continuity/association in the positive window but only a 1312 ms longest qualified run; the push-up negative stayed clean with a 464 ms longest down-like run.
- OMZ `human-pose-estimation-0005` and `0006` were measured and rejected because usable person-down evidence collapsed despite green CI. Do not retry them.
- Do not lower keypoint confidence, widen association without evidence, bridge genuine `upright`/`other`, weaken 3000 ms persistence, extend the 750 ms unknown-gap ceiling, or shop another person detector for this path.

### Broader-source Figshare generalization
Reviewed source: Figshare article `28596332`, version 2 dated 2025-03-14, reviewed license CC-BY-4.0, provenance `figshare:28596332:version-2`, archive `VideoDataset.zip` provider MD5 `c784167d08f2fa94e3afd36cec758e1f`. Exact admitted members remain ephemeral and were verified by member identity/CRC/SHA-256.

Measured unchanged-baseline pairs:
1. ACT25 ADL negative / ACT10 Fall-class positive: negative SHA-256 `7e6f026e68c280234ac34764a26b7f073e6f1367a259ed756a30a663542d3c92`, positive `8c7c13e1a9a5321e25b4203d35e65e072e2e6fcfd21ed806d7fd17b58b4438fd`; negative clean; positive 57/57 detector selections but only 32/57 safe pose associations and no candidate; about 10.02 FPS CPU.
2. ACT19 `Laying` negative / ACT4 `Fall on the back` positive: negative `1f9b3f44b67576c93a61921311830286b4a9eebe45277b90d0c9eb2625fa2a24`, positive `a54f715f3ad7d8fc2fe64390842f2c5c16ee03ace2e70f6a785cbeef6ff5f54c`; negative clean; positive 47/56 safe pose associations, 4 down and a 100 ms longest qualified run; about 9.90 FPS CPU.
3. ACT16 `Sitting` negative / ACT11 `Try to sit on chair, fall` positive: negative `3e09aa63d79a2cc3398a72ee70ea5a975f9265c29575a17af484548c95df21ac`, positive `a45ea783da1c1312880286882077c7e960bfc3830fcadf7d0b6c5a3aafe40b20`; negative clean; positive 47/57 safe pose associations and 0 down; about 15.46 FPS CPU.
4. ACT20 `Standing up from laying` negative / ACT6 `Fall on knees` positive: negative `5fe01e92aa7705a9095b5b3b6906aeeea87d34d6f72c5493c2b606f53f14ea06`, positive `a88bed7d467a0129add7e1bf2ee5dc7de4a307bd646e8a100add52798660dd63`; negative had 56/56 safe pose associations and 0 down/false alerts; positive had 55/57 safe pose associations, 12 down and a 367 ms longest qualified run; about 9.84 FPS aggregate.
5. ACT18 `Picking up` negative / ACT3 `Fall on the front` positive: negative `350587303666161ad00b812f69f64397b071e3b4d98b246556736943ca9c93f3`, positive `8ec4976d74bdf4ac046eea4946d2fd65a4b65ad834c76d02d8488da7fa5406a1`; negative clean; positive had 40/57 safe pose associations, 13 down and a 633 ms longest qualified run.

Figshare positives have clip/activity labels but no independent frame-level event interval, so they remain intentionally unscored for match/miss and alert latency. They are stage-attribution/generalization evidence only, never commercial accuracy.

## Current acceptance-moving work — platform-neutral detection + tracking
Live `main` is `5a3497b95e56180d2b7f9087ed046fb20bbb845f`, the merge of PR #60. Post-merge Analytics quality run #149 passed on attempt 1. At this session's start there were zero open implementation PRs and zero active runs for the current main head.

PR #59 established the detector-neutral normalized detection/tracked-object contract. PR #60 then landed the first portable ByteTrack-derived association slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` under MIT. It preserves the donor's high-confidence first association, low-confidence continuity association, new-track probation and bounded lost-track recovery while deliberately excluding the donor's detector/Torch path and Kalman/SciPy/LAP/`cython_bbox`/OpenCV/native-extension path. No model, weight, media, ReID or biometric identity behavior was admitted.

The current implementation branch is `feature/tracking-evaluation-metrics`. It adds a bounded platform-neutral evaluator for labeled multi-object tracking outputs. The evaluator records ground-truth observations, matched observations, misses, false-track observations, session-local ID switches, fragmentations, mean matched IoU and continuity using deterministic maximum-cardinality IoU-gated frame matching. It explicitly does not claim MOTA/HOTA benchmark parity, biometric identity, or real-world accuracy from synthetic fixtures. Focused deterministic regression tests passed 7/7 in an isolated snapshot-compatible local harness before the PR.

### First rights-cleared real tracking evidence source review
The first real multi-object/occlusion evidence candidate is `uvify-public/human_tracking_dataset` pinned to repository main commit `eb3af0cfe49de018a0c4736581daadd8eb860883`. The repository states that NCSOFT and UVify captured 500 drone videos for multi-object human tracking and provides human annotations with frame number, person ID, tracking ID, box coordinates, validity, pose, occlusion, truncation and visibility. Its published distribution reports 18,000 extracted images and 49,258 occluded object annotations. The repository license file is Creative Commons Attribution 4.0 International, copyright NCSOFT Corporation & UVify Co., Ltd.

This source is reviewed as a candidate, not yet admitted evidence. The actual dataset payload is hosted through the publisher's SharePoint link and was not accessible through the current bounded execution path, so no media/annotation hash is invented and no accuracy measurement is claimed. Before use, admit the smallest exact test sequence, record exact file/version/provenance/attribution and cryptographic hashes, keep payloads ephemeral/outside public GitHub, and map evaluation `object_id` to the dataset's tracking ID rather than person identity semantics.

The next measured comparison remains unchanged: run the retained simple-IoU association baseline and the portable ByteTrack slice on the same rights-cleared labeled multi-person sequence, then compare track fragmentation, ID switches, continuity, misses, false tracks, matched IoU, throughput/latency and resource cost. Only if that evidence localizes the largest remaining error to motion prediction or global assignment should the heavier Kalman/LAP donor path be considered.

## Pose adaptation readiness — secondary and blocked
Five measured Figshare pairs plus the GMDCSA held-out failures justify a tightly bounded pose-adaptation decision package, but training is no longer on the North Star critical path and remains blocked.

Selected trainer lineage: `Daniil-Osokin/lightweight-human-pose-estimation.pytorch@d23c284b09acf27a163e1febd511e7482cac25ed`, Apache-2.0. `analytics_lab.pose_adaptation_readiness` fails closed pending complete transitive dependency rights, authoritative cryptographic identity and artifact-specific commercial/redistribution rights for `checkpoint_iter_370000.pth`, a rights-cleared subject-separated adaptation corpus, and an exact ONNX -> OpenVINO Runtime 2026.3.1 CPU export smoke test.

GMDCSA Subjects 2-4 and all measured Figshare evidence subjects remain holdouts and must not leak into training/validation. If all gates eventually clear, the first adaptation smoke remains capped at CPU only, 2 threads, 20 wall-clock minutes, 4096 MiB RAM, 2048 MiB temporary storage, one trial and no paid compute.

## Runtime/model provenance
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Retained pose model `human-pose-estimation-0001` FP16.
- Retained evidence detector `person-detection-0200` FP16.
- OpenVINO Runtime `2026.3.1`.
- Evidence decoder `opencv-python-headless==4.12.0.88`.
- Corrected OpenPose decoder pinned to the same OMZ revision with attribution preserved.
- Portable ByteTrack association donor revision `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, MIT, adapted code only with no donor model/weights/data or new runtime dependency.
- Trainer candidate `Daniil-Osokin/lightweight-human-pose-estimation.pytorch@d23c284b09acf27a163e1febd511e7482cac25ed`, candidate only; training not authorized.

## Evidence/data provenance
### GMDCSA-24
Repository `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`, pinned revision `5abac7693229900cf80f722e878fbb119211fc1c`, reviewed repository license MIT, Zenodo DOI `10.5281/zenodo.13354453`, paper DOI `10.1016/j.dib.2024.110892`.

### Figshare 2017-activity source
Article `28596332`, version 2, reviewed license CC-BY-4.0, provenance `figshare:28596332:version-2`, activity mapping reference DOI `10.30970/eli.33.12` under CC-BY-4.0. Media remains outside public GitHub.

### UVify/NCSOFT human tracking source candidate
Repository `uvify-public/human_tracking_dataset`, pinned revision `eb3af0cfe49de018a0c4736581daadd8eb860883`, published license CC-BY-4.0. Exact data payload not yet admitted or hashed; do not treat the repository metadata review as dataset accuracy evidence.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. This session began from live `main` `5a3497b95e56180d2b7f9087ed046fb20bbb845f`, zero open implementation PRs, zero active runs for that head, zero unchanged retries, zero CI dispatches and zero consecutive stalled sessions. Post-merge main run #149 was green on attempt 1.

Live repository/head/open-PR/CI state and the unchanged efficiency-policy ceilings were verified before mutation. No additional tracker candidates were shopped. Branch `feature/tracking-evaluation-metrics` was created from the exact live main. Focused tracking-evaluator regression tests passed 7/7 locally; these are code-regression evidence only, not tracking-accuracy evidence. Opening the implementation PR is the first CI-triggering action in this session; no retry has been consumed.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Focused tracker/evaluator regressions after checkout:
```sh
python -m unittest tests.test_bytetrack tests.test_tracking_evaluation -v
```

Pose adaptation readiness can still be inspected locally without training or network access:
```sh
python - <<'PY'
from analytics_lab.pose_adaptation_readiness import training_blockers
print("\n".join(training_blockers()))
PY
```

## Next executable decision
Open exactly one implementation PR from `feature/tracking-evaluation-metrics` on unchanged base `5a3497b95e56180d2b7f9087ed046fb20bbb845f`. Require that exact head to pass Linux, Windows and the Analytics quality gate. Merge only if the tested head/base are unchanged and all required checks are green.

After the evaluator lands, obtain the smallest exact rights-cleared labeled tracking sequence from the reviewed UVify/NCSOFT source or another already-approved source without expanding donor shopping. Bind exact payload hashes/provenance, keep data ephemeral, and run the unchanged simple-IoU versus portable ByteTrack comparison. Do not add Kalman/LAP/native extensions unless measured evidence demonstrates their missing capability is the largest error source.

## Outstanding commercial-release gates
For detection/tracking: substantially broader held-out real-video evidence across people/vehicles/objects, crowded scenes, crossings, occlusions, low light, viewpoints and resolutions; defensible detection precision/recall where labels permit; track fragmentation and ID-switch measurements; throughput/latency/resource envelope; Linux/Windows portability; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.

For later priorities, LPR/OCR, Face (including selectable face blurring), Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before any commercial claim.
