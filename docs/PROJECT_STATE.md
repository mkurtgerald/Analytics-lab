# Project state — 2026-09-18

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

### Product priority
1. Detection + tracking — North Star foundation.
2. LPR/OCR.
3. Face.
4. Weapons.
5. Appearance search.

Person-down/slip-fall remains retained and important but is secondary. Continue it only where it validates shared perception, tracking, evidence, or evaluation primitives, or when a higher-priority path is concretely blocked.

## Retained person-down path — secondary validation analytic
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media/model/runtime identities and aggregate measurements while source media remains ephemeral and outside public GitHub.

## Accepted person-down evidence frontier
The retained detector/temporal path is stable; the repeated limiting boundary is pose representation/association/posture continuity. This evidence remains useful to shared perception and evaluation, but it no longer owns the product critical path.

### GMDCSA-24 held-out evidence
- Subject 1 seed/untouched rotation: positives legitimately emitted while hard negatives remained clean; untouched positive alert delay was 3992 ms, detector coverage 150/150, safe pose associations 175/181 overall, 107 qualified down observations and a 4096 ms longest qualified run. This is engineering evidence, not commercial accuracy.
- Subject 2: orientation recovery improved detector coverage and usable down evidence, but the positive remains a scored miss. A later untouched night/cross-view positive had 76/82 detector selections but only 32/82 safe pose associations and a 736 ms longest qualified run; its negative had 226/226 safe pose associations and 0 false alerts.
- Subject 3: unchanged retained path produced 0 candidates / 1 miss on the positive and 0 false alerts on the sitting-to-sleeping negative. Selected-person crop fallback improved association but regressed usable down continuity and was rejected.
- Subject 4: detector selection, continuity and safe pose association were 129/129 in the positive fall window, yet longest qualified down run was only 1312 ms. Dynamic push-up negative remained clean with a 464 ms longest down-like run. This strongly argues against weakening the 3000 ms persistence rule.

Subjects 2, 3 and 4 remain protected evaluation holdouts and must not be used for training/adaptation merely to make prior misses pass.

### Rejected pose alternatives
- OMZ `human-pose-estimation-0005`: rejected after usable person-down evidence collapsed to 1 qualified down / 0 ms.
- OMZ `human-pose-estimation-0006`: rejected after the same collapse; 102 labeled-positive observations failed the unchanged required-keypoint-confidence floor despite higher throughput.

Do not retry `0005`/`0006`, lower required-keypoint confidence, widen association without evidence, bridge genuine `upright`/`other`, weaken 3000 ms persistence, extend the 750 ms unknown-gap ceiling, or shop another person detector for the person-down path.

### Broader-source Figshare generalization
Reviewed source:
- article `28596332`, version 2 dated 2025-03-14;
- source id `figshare-fall-2017-activities`, reviewed license `CC-BY-4.0`, provenance `figshare:28596332:version-2`;
- archive `VideoDataset.zip`, 2,529,520,868 bytes, provider MD5 `c784167d08f2fa94e3afd36cec758e1f`;
- 22,397 classic-ZIP entries, 20,324 non-directory members, 2,022 bounded MP4 entries;
- central directory fetched in bounded ranges only; member media is admitted by exact range, CRC and SHA-256 and remains ephemeral.

Measured unchanged-baseline pairs:
1. ACT25 ADL negative / ACT10 Fall-class positive: negative SHA-256 `7e6f026e68c280234ac34764a26b7f073e6f1367a259ed756a30a663542d3c92`, positive `8c7c13e1a9a5321e25b4203d35e65e072e2e6fcfd21ed806d7fd17b58b4438fd`; negative 56/56 detector and pose associations, all upright, 0 false alerts; positive 57/57 detector selections but only 32/57 safe pose associations, 1 down / 4 other / 32 unknown / 20 upright, no candidate; about 10.02 FPS CPU.
2. ACT19 `Laying` negative / ACT4 `Fall on the back` positive: negative SHA-256 `1f9b3f44b67576c93a61921311830286b4a9eebe45277b90d0c9eb2625fa2a24`, positive `a54f715f3ad7d8fc2fe64390842f2c5c16ee03ace2e70f6a785cbeef6ff5f54c`; negative clean with 56/56 safe pose associations and 0 false alerts; positive 47/56 safe pose associations, 39 upright / 9 unknown / 4 other / 4 down, longest qualified run 100 ms, no candidate; unmatched poses were not borderline associations; about 9.90 FPS CPU.
3. ACT16 `Sitting` negative / ACT11 `Try to sit on chair, fall` positive: negative SHA-256 `3e09aa63d79a2cc3398a72ee70ea5a975f9265c29575a17af484548c95df21ac`, positive `a45ea783da1c1312880286882077c7e960bfc3830fcadf7d0b6c5a3aafe40b20`; negative 56/56 detector and safe pose associations, all upright, 0 false alerts; positive 57/57 detector selections but 47/57 safe pose associations, 35 upright / 19 unknown / 3 other / 0 down, no candidate; about 15.46 FPS CPU.
4. ACT20 `Standing up from laying` negative / ACT6 `Fall on knees` positive: negative SHA-256 `5fe01e92aa7705a9095b5b3b6906aeeea87d34d6f72c5493c2b606f53f14ea06`, positive `a88bed7d467a0129add7e1bf2ee5dc7de4a307bd646e8a100add52798660dd63`. The difficult floor-transition negative had 56/56 detector selections, 55/55 continuity links and 56/56 safe pose associations, with 26 upright / 25 other / 5 unknown / 0 down, a 0 ms longest qualified down run, 0 candidates and 0 false alerts. The positive had 57/57 detector selections, 56/56 continuity links and 55/57 safe pose associations, with 37 upright / 12 down / 5 unknown / 3 other; its 12 qualified down observations formed only a 367 ms longest qualified run, with 3 bounded unknown frames bridged over at most 100 ms and no candidate. Aggregate throughput was about 9.84 FPS over 113 frames.
5. ACT18 `Picking up` negative / ACT3 `Fall on the front` positive: negative SHA-256 `350587303666161ad00b812f69f64397b071e3b4d98b246556736943ca9c93f3`, positive SHA-256 `8ec4976d74bdf4ac046eea4946d2fd65a4b65ad834c76d02d8488da7fa5406a1`. The negative had 56/56 detector selections, 55/55 continuity links, 56/56 safe pose associations, 0 down observations, 0 candidates and 0 false alerts. The positive had 57/57 detector selections, 56/56 continuity links, 40/57 safe pose associations, 13 down / 24 unknown / 20 upright observations and a 633 ms longest qualified down run, below the unchanged 3000 ms persistence boundary.

Figshare positives have only source clip/activity classes; without an independent frame-level interval they are intentionally unscored for match/miss and alert delay. These short pairs are stage-attribution/generalization evidence only, never commercial accuracy.

Pair 5 was merged by PR #58 on `16509759ed24284bf7d2d11346886ab6dc2b2ad2`. Exact-head run #144 and post-merge main run #145 passed on attempt 1. No detector, pose, association, posture, persistence, unknown-gap threshold, training plan, paid resource, home/customer media or commercial-accuracy claim changed.

## Current acceptance-moving work — platform-neutral detection + tracking
Live `main` `16509759ed24284bf7d2d11346886ab6dc2b2ad2` is green with zero open implementation PRs at the start of this work item. The highest-priority unmet boundary is a detector-neutral multi-object tracking contract. The existing `IoUTracker` remains a useful deterministic person/pose baseline, but it is not intended to become the North Star commodity tracker.

Donor review is capped at three candidates and stops here:
- `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, MIT: selected as the first integration target because its two-stage high/low-confidence association directly matches the measured continuity need while keeping identity/ReID separate. Do not vendor or admit its implementation yet; first pin the exact donor slice, transitive dependency rights/notices, and Linux/Windows CPU compatibility.
- `tryolabs/norfair@e517b4236f6b67a6ecf342f5df1fccb7788dbc54`, BSD-3-Clause: detector-agnostic, production-stable fallback; its `filterpy`/`rich`/`scipy`/`numpy` dependency surface is larger and still requires transitive rights/runtime review.
- Open Model Zoo at `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0: its multi-camera multi-target tracking demo is retained as a reference and possible later appearance-search/ReID donor, but it is heavier and ReID/model-coupled, so it is not the first single-camera tracking integration.

This implementation item adds only a platform-neutral normalized detection/tracked-object contract plus a bounded session wrapper. It adds no tracker algorithm, external runtime, model, weights, media, biometric identity, ReID, or accuracy claim. The immediate next step after this boundary is exact donor-slice/dependency review for ByteTrack and then a small adapter behind this contract, followed by measured comparison against the current IoU baseline on rights-cleared multi-person/crossing/occlusion evidence.

## Pose adaptation readiness — secondary and blocked pending prerequisites
Five independent measured Figshare pairs plus the GMDCSA held-out failures localize the dominant person-down error source strongly enough to justify a tightly bounded pose adaptation decision package, but training remains blocked and is no longer on the North Star critical path.

The selected first trainer lineage is `Daniil-Osokin/lightweight-human-pose-estimation.pytorch` at exact revision `d23c284b09acf27a163e1febd511e7482cac25ed`, Apache-2.0. `analytics_lab.pose_adaptation_readiness` fails closed before any training. Current blockers are complete transitive dependency rights, authoritative cryptographic identity plus artifact-specific commercial/redistribution rights for `checkpoint_iter_370000.pth`, a rights-cleared subject-separated adaptation corpus, and an exact ONNX -> OpenVINO Runtime 2026.3.1 CPU export smoke test.

GMDCSA Subjects 2-4 and all measured Figshare evidence subjects remain evaluation holdouts and must not leak into training or validation. If every gate clears and the higher-priority work permits it, the first adaptation smoke is capped at CPU only, 2 threads, 20 wall-clock minutes, 4096 MiB RAM, 2048 MiB temporary storage, one trial and no paid compute. This is a compatibility/learning-signal ceiling, not an accuracy or production claim.

## Runtime/model provenance
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0;
- retained pose model `human-pose-estimation-0001` FP16;
- retained evidence detector `person-detection-0200` FP16;
- OpenVINO Runtime `2026.3.1`;
- evidence decoder `opencv-python-headless==4.12.0.88`;
- corrected OpenPose decoder pinned to the same OMZ revision with attribution preserved;
- trainer candidate `Daniil-Osokin/lightweight-human-pose-estimation.pytorch@d23c284b09acf27a163e1febd511e7482cac25ed`, candidate only, training not authorized.

## Evidence/data provenance
### GMDCSA-24
Repository `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`, pinned revision `5abac7693229900cf80f722e878fbb119211fc1c`, reviewed repository license MIT, Zenodo DOI `10.5281/zenodo.13354453`, paper DOI `10.1016/j.dib.2024.110892`.

### Figshare 2017-activity source
Article `28596332`, version 2, reviewed license CC-BY-4.0, provenance `figshare:28596332:version-2`, activity mapping reference DOI `10.30970/eli.33.12` under CC-BY-4.0. Media remains outside public GitHub.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. This session began from live `main` `16509759ed24284bf7d2d11346886ab6dc2b2ad2`, zero open implementation PRs, zero active runs for main, zero unchanged retries, zero CI dispatches and zero consecutive stalled sessions. Post-merge main run #145 was green on attempt 1. The machine preflight allowed implementation.

Branch `feature/donor-tracker-boundary` was created from the exact live main. Donor research stopped after exactly three candidates. Focused local contract tests passed 5/5 before the branch update. Opening the implementation PR is the first CI-triggering action in this session; no unchanged retry has been consumed.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Pose adaptation readiness can still be inspected locally without training or network access:
```sh
python - <<'PY'
from analytics_lab.pose_adaptation_readiness import training_blockers
print("\n".join(training_blockers()))
PY
```

## Next executable decision
Require the exact head of `feature/donor-tracker-boundary` to pass Linux, Windows and the Analytics quality gate on unchanged base `16509759ed24284bf7d2d11346886ab6dc2b2ad2`. Merge only if that exact head is green.

After the boundary lands, continue the same highest-priority detection/tracking item by pinning the smallest ByteTrack donor slice and its transitive dependency/license/notices/runtime compatibility before copying any donor implementation. Then implement the smallest adapter behind `analytics_lab.tracking` and measure it against the existing IoU baseline on rights-cleared multi-person/crossing/occlusion evidence. Reject the donor path if it cannot produce a measurable tracking improvement without unacceptable portability, rights, or resource regression.

## Outstanding commercial-release gates
For detection/tracking: substantially broader held-out real-video evidence across people/vehicles/objects, crowded scenes, crossings, occlusions, low light, viewpoints and resolutions; defensible detection precision/recall where labels permit; track fragmentation and ID-switch measurements; throughput/latency/resource envelope; Linux/Windows portability; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.

For later priority analytics, LPR/OCR, Face, Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before any commercial claim.
