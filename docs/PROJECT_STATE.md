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

## Detection + tracking — current North Star work
Live `main` at this work item's start is `a79fdb15a7d9eaf189c662b0d46cd0b027066803`, the merge of PR #65. Post-merge Analytics quality run #160 passed on attempt 1. At start there were zero open implementation PRs and zero queued/running runs for that exact head.

Landed boundaries:
- PR #59: detector-neutral normalized detection/tracked-object contract.
- PR #60: portable ByteTrack-derived association slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, MIT. It preserves high-confidence then low-confidence association while excluding the heavier Torch/Kalman/SciPy/LAP/native-extension/model stack pending measured need.
- PR #61: bounded platform-neutral tracking evaluator for matches, misses, false-track observations, session-local ID switches, fragmentations, matched IoU and continuity. Synthetic fixtures are regression evidence only.
- PR #62: fail-closed UVify/NCSOFT tracking-annotation parser pinned to `uvify-public/human_tracking_dataset@eb3af0cfe49de018a0c4736581daadd8eb860883`; source tracking IDs are evaluation identifiers, not identities.
- PR #63: tracking-evidence admission requiring exhaustive multi-object annotation scope, annotation SHA-256, ordered frame-manifest SHA-256, frame count and dimensions before multi-object metrics are accepted.
- PR #64: evidence-byte rights admission requiring an authoritative rights source and explicit commercial-evaluation authorization. BDD100K was rejected from the current commercial path because its dataset terms do not establish general commercial rights for Analytics Lab.
- PR #65: exact CC0 Wikimedia pedestrian source admitted fail-closed on byte size, published SHA-1 and pinned SHA-256; no media retained, no tracker/model run and no accuracy claim.

## Current acceptance item — pre-register the first real tracker-comparison window
The next comparison must not cherry-pick an easy interval after seeing detector/tracker output. The first measured window is therefore fixed in code before benchmark output inspection.

Selected source: Wikimedia Commons `Video Codec Test pedestrian area 1080p25.y4m.webm`.

Pinned source/rights record:
- author: Taurus Media Technik;
- license: CC0-1.0;
- rights page: `https://commons.wikimedia.org/w/index.php?title=File:Video_Codec_Test_pedestrian_area_1080p25.y4m.webm&oldid=1196486240`;
- canonical upload URL: `https://upload.wikimedia.org/wikipedia/commons/a/ae/Video_Codec_Test_pedestrian_area_1080p25.y4m.webm`;
- published size: 11,215,394 bytes;
- published SHA-1: `51e89a672896e45cca17aa46cd223630a6266e26`;
- pinned SHA-256: `bfadaa62cccb42db875d50bb842aa0964fbf72040432e4097c1df59e043e0c26`;
- duration/dimensions: 15.125 seconds, 1920x1080, nominal 25 fps.

Pre-registered smoke window:
- source frames: **150-174 inclusive**;
- frame count: **25**;
- time span at 25 fps: **6.00s through 6.96s**;
- measured class: **person**;
- purpose: first real-video simple-IoU-vs-portable-ByteTrack smoke comparison;
- canonical benchmark-plan SHA-256: `eb7995a389a22f3528b9bfca97d64f9d8e3c515ebf1c764cf25b68fbdcc3479c`.

`analytics_lab.tracking_benchmark_plan` validates and hashes the source identity, exact frame window, frame rate, dimensions, class, purpose and selection basis. The fixed helper intentionally describes this as acceptance/smoke evidence, not commercial accuracy. Focused local regressions pass 3/3: exact frame selection/digest, deterministic hash binding, and fail-closed invalid/unbounded plan inputs.

### Next measured comparison
After this exact pre-registration head is green and merged:
1. extract only frames 150-174 ephemerally from the already-admitted source;
2. create independent exhaustive person boxes/session-local ground-truth IDs for every measured frame without using benchmarked detector/tracker outputs as truth;
3. bind annotation SHA-256 and ordered per-frame SHA-256 manifest through `TrackingEvidenceManifest`;
4. run the unchanged simple-IoU baseline and portable ByteTrack slice on identical detector observations;
5. compare raw fragmentation, ID switches, continuity, misses, false tracks, matched IoU, throughput/latency and CPU/resource cost;
6. attack only the largest demonstrated error source;
7. expand to a larger/crowded/occlusion window only after the smoke comparison is measured and recorded.

Do not admit Kalman/LAP/native extensions, another tracker donor, training, or parameter tuning until this comparison demonstrates a concrete need.

## Rejected/blocked tracking evidence sources retained for provenance
- **UVify/NCSOFT**: pinned revision `eb3af0cfe49de018a0c4736581daadd8eb860883`, published CC-BY-4.0, useful exhaustive-style tracking annotations, but the publisher-hosted payload remains unavailable through the bounded path; no sequence bytes or metrics admitted.
- **D-PTUAC**: CC-BY-4.0 but single-target visual-object tracking, not exhaustive multi-object ground truth; roughly 15 GB package intentionally not downloaded.
- **BDD100K**: technically suitable for MOT, but its data/label terms do not establish general commercial product-development rights for this project without qualifying affiliation or separate licensing; no dataset media downloaded.

## Retained person-down path — required secondary analytic
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. The detector/temporal path remains stable; remaining misses localize primarily to pose representation/association/posture continuity. OMZ `human-pose-estimation-0005` and `0006` were measured and rejected. Do not weaken persistence, confidence, upright/other reset behavior, or holdouts merely to improve apparent results.

## Runtime/model provenance retained
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Retained pose model `human-pose-estimation-0001` FP16.
- Retained evidence detector `person-detection-0200` FP16.
- OpenVINO Runtime `2026.3.1`.
- Evidence decoder `opencv-python-headless==4.12.0.88`.
- Portable ByteTrack donor revision `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, MIT; adapted association code only, no donor model/weights/data or new runtime dependency.

## Evidence/data provenance retained
- **GMDCSA-24**: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos@5abac7693229900cf80f722e878fbb119211fc1c`, reviewed repository license MIT, Zenodo DOI `10.5281/zenodo.13354453`, paper DOI `10.1016/j.dib.2024.110892`.
- **Figshare 2017 activity source**: article `28596332`, version 2, CC-BY-4.0, provenance `figshare:28596332:version-2`; media remains outside public GitHub.
- **Wikimedia CC0 pedestrian source**: exact rights page/upload URL above; exact byte size, SHA-1 and SHA-256 pinned above.

## Efficiency ledger
One worker, one acceptance-moving work item. This session began with zero open implementation PRs, zero queued/running runs for live `main`, zero unchanged retries, zero CI-triggering actions and zero stalled sessions. Fresh local guardrail preflight returned `{"allowed": true, "reasons": []}`. Branch-only commits do not trigger CI; opening the implementation PR will be CI-triggering action 1/2. No media/model download, paid resource, self-hosted runner, training, threshold change, identity behavior or accuracy claim is introduced by this item.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Focused Detection + Tracking regressions:
```sh
python -m unittest tests.test_bytetrack tests.test_tracking_evaluation tests.test_uvify_tracking tests.test_tracking_evidence tests.test_wikimedia_tracking_admission tests.test_tracking_benchmark_plan -v
```

## Merge rule
Merge this acceptance item only if its exact current head passes Linux, Windows and the Analytics quality gate on unchanged base `a79fdb15a7d9eaf189c662b0d46cd0b027066803`. Verify base/head immediately before merge. One unchanged retry is permitted only for a diagnosed transient infrastructure failure; deterministic failure requires a fix before another run.

## Outstanding commercial-release gates
Detection/tracking still requires substantially broader held-out real-video evidence across people/vehicles/objects, crowded scenes, crossings, occlusions, low light, viewpoints and resolutions; defensible precision/recall where labels permit; track fragmentation/ID-switch measurements; throughput/latency/resource envelopes; Linux/Windows portability; privacy/security/provenance and notices review; versioned installable integration; packaging; and explicit owner commercial-release approval.

LPR/OCR, Face (including selectable face blurring), Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before any commercial claim.
