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
- source id `figshare-fall-2017-activities`;
- reviewed license `CC-BY-4.0`;
- provenance `figshare:28596332:version-2`;
- archive `VideoDataset.zip`, 2,529,520,868 bytes, provider MD5 `c784167d08f2fa94e3afd36cec758e1f`;
- 22,397 classic-ZIP entries, 20,324 non-directory members, 2,022 bounded MP4 entries;
- central directory fetched in bounded ranges only; member media is admitted by exact range, CRC and SHA-256 and remains ephemeral.

Measured unchanged-baseline pairs:

1. ACT25 ADL negative / ACT10 Fall-class positive
   - negative SHA-256 `7e6f026e68c280234ac34764a26b7f073e6f1367a259ed756a30a663542d3c92`;
   - positive SHA-256 `8c7c13e1a9a5321e25b4203d35e65e072e2e6fcfd21ed806d7fd17b58b4438fd`;
   - negative: 56/56 detector selections, 56/56 safe pose associations, 56 upright, 0 candidates / 0 false alerts;
   - positive: 57/57 detector selections, 56/56 linked transitions, only 32/57 safe pose associations, posture 1 down / 4 other / 32 unknown / 20 upright, no candidate;
   - pair throughput about 10.02 FPS CPU.

2. ACT19 `Laying` negative / ACT4 `Fall on the back` positive
   - negative SHA-256 `1f9b3f44b67576c93a61921311830286b4a9eebe45277b90d0c9eb2625fa2a24`;
   - positive SHA-256 `a54f715f3ad7d8fc2fe64390842f2c5c16ee03ace2e70f6a785cbeef6ff5f54c`;
   - negative: 56/56 detector selections, 56/56 safe pose associations, 0 down, 0 candidates / 0 false alerts;
   - positive: 56/56 detector selections, 47/56 safe pose associations, posture 39 upright / 9 unknown / 4 other / 4 down, longest qualified run 100 ms, no candidate;
   - unmatched poses were not borderline associations; widening association is not justified;
   - pair throughput about 9.90 FPS CPU.

3. ACT16 `Sitting` negative / ACT11 `Try to sit on chair, fall` positive
   - negative SHA-256 `3e09aa63d79a2cc3398a72ee70ea5a975f9265c29575a17af484548c95df21ac`;
   - positive SHA-256 `a45ea783da1c1312880286882077c7e960bfc3830fcadf7d0b6c5a3aafe40b20`;
   - negative: 56/56 detector selections, 55/55 continuity links, 56/56 safe pose associations, 56 upright, 0 down, 0 candidates / 0 false alerts;
   - positive: 57/57 detector selections, 56/56 continuity links, 47/57 safe pose associations, posture 35 upright / 19 unknown / 3 other / 0 down, no candidate;
   - pair throughput about 15.46 FPS CPU.

Figshare positives have only source clip/activity classes; without an independent frame-level interval they are intentionally unscored for match/miss and alert delay. These three short pairs are stage-attribution/generalization evidence only, never commercial accuracy.

## Current acceptance-moving work — pose adaptation readiness
Three independent Figshare pairs plus the GMDCSA held-out failures now localize the dominant error source strongly enough to justify a tightly bounded pose adaptation decision package.

The selected first trainer lineage is `Daniil-Osokin/lightweight-human-pose-estimation.pytorch` at exact revision `d23c284b09acf27a163e1febd511e7482cac25ed`, Apache-2.0. Upstream directly documents the Lightweight OpenPose training path used by the retained model family and the ONNX/OpenVINO export chain.

`analytics_lab.pose_adaptation_readiness` records the candidate and fails closed before any training. Current blockers are:
- complete transitive commercial/redistribution rights for the historical dependency stack are not yet verified;
- upstream `checkpoint_iter_370000.pth` source is identified but no authoritative cryptographic hash is yet pinned;
- artifact-specific commercial/redistribution rights for that checkpoint are unresolved;
- no rights-cleared adaptation corpus with explicit subject-separated train and validation identities is admitted;
- exact ONNX -> OpenVINO Runtime 2026.3.1 CPU export compatibility has not yet been smoke-tested.

The already-measured Figshare subjects SBJ_01, SBJ_10, SBJ_06, SBJ_03, SBJ_02 and SBJ_09 are also locked as evaluation holdouts. They must not leak into training or validation.

If all legal/data/export gates clear, the first adaptation smoke is capped at CPU only, 2 threads, 20 wall-clock minutes, 4096 MiB RAM, 2048 MiB temporary storage, one trial and no paid compute. This is only a compatibility/learning-signal ceiling, not an accuracy or production claim.

See `docs/donor-review-lightweight-openpose-training.md` for the exact candidate review.

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
- repository `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`;
- pinned revision `5abac7693229900cf80f722e878fbb119211fc1c`;
- reviewed repository license MIT;
- Zenodo DOI `10.5281/zenodo.13354453`;
- paper DOI `10.1016/j.dib.2024.110892`.

### Figshare 2017-activity source
- article `28596332`, version 2;
- reviewed license CC-BY-4.0;
- provenance `figshare:28596332:version-2`;
- activity mapping reference DOI `10.30970/eli.33.12`, CC-BY-4.0;
- media remains outside public GitHub.

## Efficiency ledger
One worker, one acceptance-moving work item and at most one implementation PR. Intake for this cycle verified live `main` at `62db3c9a89d54d51a0ef81f95fb21935bda37950`, zero open PRs, zero active runs for the head, and exact-head post-merge run #132 green on attempt 1. The local preflight snapshot allowed implementation with zero unchanged retries, zero CI dispatches this session and zero sessions without progress.

This cycle adds one fail-closed readiness module, its focused tests, the upstream review and this state update. It downloads no model/media, trains nothing, changes no detector/pose/posture/association/temporal threshold, uses no paid resource and makes no commercial-accuracy claim.

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
Do not train yet. First resolve the exact `checkpoint_iter_370000.pth` cryptographic identity and artifact-specific commercial/redistribution right from an authoritative upstream source, or reject that initialization path. In parallel, admit a rights-cleared adaptation corpus with explicit subject-separated train/validation/holdout identities. Then smoke-test the exact pinned trainer export through ONNX into OpenVINO Runtime 2026.3.1 CPU inside the bounded resource envelope. Only if every machine-enforced readiness blocker clears may one bounded adaptation trial run.

Continue expanding untouched broader-source evidence while those rights/data prerequisites are resolved; do not tune around held-out Subjects 2-4 or any already-measured Figshare subject.

## Outstanding commercial-release gates
Substantially broader held-out positive/negative real-video evidence across different subjects, cameras, sites, resolutions, viewpoints, lighting and multi-person scenes; meaningful false-alert/camera-hour and missed-event measurements; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.
