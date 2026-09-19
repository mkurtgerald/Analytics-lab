# Project state — 2026-09-18

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Retained person-down path
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media/model/runtime identities and aggregate measurements while source media remains ephemeral and outside public GitHub.

## Accepted evidence frontier
The retained detector/temporal path is stable; the repeated limiting boundary is pose representation/association/posture continuity.

### GMDCSA-24 held-out evidence
- Subject 1 seed/untouched rotation: positives legitimately emitted while hard negatives remained clean; untouched positive alert delay was 3992 ms, detector coverage 150/150, safe pose associations 175/181 overall, 107 qualified down observations and a 4096 ms longest qualified run. This is engineering evidence, not commercial accuracy.
- Subject 2: orientation recovery improved detector coverage and usable down evidence, but the positive remains a scored miss. A later untouched night/cross-view positive had 76/82 detector selections but only 32/82 safe pose associations and a 736 ms longest qualified run; its negative had 226/226 safe pose associations and 0 false alerts.
- Subject 3: unchanged retained path produced 0 candidates / 1 miss on the positive and 0 false alerts on the sitting-to-sleeping negative. Selected-person crop fallback improved association but regressed usable down continuity and was rejected.
- Subject 4: detector selection, continuity and safe pose association were 129/129 in the positive fall window, yet longest qualified down run was only 1312 ms. Dynamic push-up negative remained clean with a 464 ms longest down-like run. This strongly argues against weakening the 3000 ms persistence rule.

Subjects 2, 3 and 4 remain protected evaluation holdouts and must not be used for training/adaptation merely to make prior misses pass.

### Rejected pose alternatives
- OMZ `human-pose-estimation-0005`: rejected after usable person-down evidence collapsed to 1 qualified down / 0 ms.
- OMZ `human-pose-estimation-0006`: rejected after the same collapse; 102 labeled-positive observations failed the unchanged required-keypoint-confidence floor despite higher throughput.

Do not retry `0005`/`0006`, lower required-keypoint confidence, widen association without evidence, bridge genuine `upright`/`other`, weaken 3000 ms persistence, extend the 750 ms unknown-gap ceiling, or shop another person detector.

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
4. ACT20 `Standing up from laying` negative / ACT6 `Fall on knees` positive: negative SHA-256 `5fe01e92aa7705a9095b5b3b6906aeeea87d34d6f72c5493c2b606f53f14ea06`, positive `a88bed7d467a0129add7e1bf2ee5dc7de4a307bd646e8a100add52798660dd63`. Preserved first-head run #138 measured the unchanged path. The difficult floor-transition negative had 56/56 detector selections, 55/55 continuity links and 56/56 safe pose associations, with 26 upright / 25 other / 5 unknown / 0 down, a 0 ms longest qualified down run, 0 candidates and 0 false alerts. The positive had 57/57 detector selections, 56/56 continuity links and 55/57 safe pose associations, with 37 upright / 12 down / 5 unknown / 3 other; its 12 qualified down observations formed only a 367 ms longest qualified run, with 3 bounded unknown frames bridged over at most 100 ms and no candidate. Aggregate throughput was about 9.84 FPS over 113 frames. This positive is deliberately unscored for match/miss and alert delay because the source does not provide a defensible frame-level event interval.

Figshare positives have only source clip/activity classes; without an independent frame-level interval they are intentionally unscored for match/miss and alert delay. These short pairs are stage-attribution/generalization evidence only, never commercial accuracy.

Pair 4 sharpens the same architectural conclusion without just repeating the earlier association failure: detection and continuity remained complete, and pose association was nearly complete on the positive, yet usable down posture survived for only 367 ms. That points to pose representation/posture continuity over the event rather than the detector or a justification to weaken the 3000 ms persistence rule. The hard floor-transition negative staying clean reinforces the conservative temporal boundary.

## Current acceptance-moving work — broader generalization pair 4
PR #55 selected ACT20 `Standing up from laying` against ACT6 `Fall on knees` before any member payload/model output was inspected. Exact-head selector run #136 chose:
- negative: SBJ_29 / LOC1, `VideoDataset/ADL/SBJ_29_LOC1/ACT20_R_1/20240921145245.mp4`, compressed 580,669 bytes, uncompressed 581,347 bytes, CRC32 `67dbcbe6`, local-header offset 1,118,347,300;
- positive: SBJ_07 / LOC3, `VideoDataset/Fall/SBJ_07_LOC3/ACT6_R_1/20240914124502.mp4`, compressed 518,570 bytes, uncompressed 519,269 bytes, CRC32 `c6d0fb47`, local-header offset 1,869,305,105.

PR #56 first exact evidence head `41ca8ddb6db36571b104a769061f61be32595bc2`, run #138, passed on attempt 1 across Linux, Windows, 43 guardrails, 272 regression tests (one optional OpenCV skip), the bounded archive probe, unchanged CPU evidence and the Analytics quality gate against unchanged base `dc9c217f1a6c1307a533fc581fa412f1f8e4bc45`. It discovered the exact SHA-256 identities above and retained no media. Those hashes are now pinned fail-closed; the second/final CI-triggering head must reproduce the same semantic evidence before merge.

## Pose adaptation readiness — blocked pending prerequisites
Four independent measured Figshare pairs plus the GMDCSA held-out failures localize the dominant error source strongly enough to justify a tightly bounded pose adaptation decision package, but training remains blocked.

The selected first trainer lineage is `Daniil-Osokin/lightweight-human-pose-estimation.pytorch` at exact revision `d23c284b09acf27a163e1febd511e7482cac25ed`, Apache-2.0. `analytics_lab.pose_adaptation_readiness` fails closed before any training. Current blockers are complete transitive dependency rights, authoritative cryptographic identity plus artifact-specific commercial/redistribution rights for `checkpoint_iter_370000.pth`, a rights-cleared subject-separated adaptation corpus, and an exact ONNX -> OpenVINO Runtime 2026.3.1 CPU export smoke test.

GMDCSA Subjects 2-4 and all measured/selected Figshare evidence subjects, including SBJ_29 and SBJ_07, are evaluation holdouts and must not leak into training or validation. If every gate clears, the first adaptation smoke is capped at CPU only, 2 threads, 20 wall-clock minutes, 4096 MiB RAM, 2048 MiB temporary storage, one trial and no paid compute. This is a compatibility/learning-signal ceiling, not an accuracy or production claim.

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
One worker, one acceptance-moving work item, one implementation PR. This bounded session began from live `main` `dc9c217f1a6c1307a533fc581fa412f1f8e4bc45`, zero open implementation PRs, zero active runs for main, zero unchanged retries and zero stalled sessions. PR #56 is the sole implementation PR. Its first CI-triggering head consumed one of two allowed dispatches and run #138 passed on attempt 1; no unchanged retry was used. The current hash-pin commit is the second and final CI-triggering head permitted in this session.

No detector, pose, association, posture, 3000 ms persistence or 750 ms unknown-gap threshold changes are in this work item. No training, paid resources, home/customer media, retained public media, licensing shortcut or commercial-accuracy claim is permitted.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Pose adaptation readiness can be inspected locally without training or network access:
```sh
python - <<'PY'
from analytics_lab.pose_adaptation_readiness import training_blockers
print("\n".join(training_blockers()))
PY
```

## Next executable decision
Run the second/final exact-head CI for PR #56 with both media SHA-256 values pinned. Require the same admitted bytes and semantic stage evidence, exact-head green Linux and Windows regressions, and the Analytics quality gate on unchanged base. If it passes, merge PR #56; if it fails deterministically, preserve the first failure and stop CI work for this session rather than exceeding the two-dispatch budget.

Preserve the 3000 ms persistence rule, 750 ms unknown-gap ceiling and every existing detector/pose/association/posture threshold. Do not train yet.

## Outstanding commercial-release gates
Substantially broader held-out positive/negative real-video evidence across different subjects, cameras, sites, resolutions, viewpoints, lighting and multi-person scenes; meaningful false-alert/camera-hour and missed-event measurements; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.
