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
Live `main` at this work item's start is `04b8c491432fb2850291d2d4ecf06470ce842ce1`, the merge of PR #64. Post-merge Analytics quality run #157 passed on attempt 1. At start there were zero open implementation PRs and zero queued/running runs for that exact head.

Landed boundaries:
- PR #59: detector-neutral normalized detection/tracked-object contract.
- PR #60: portable ByteTrack-derived association slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, MIT. It keeps high-confidence then low-confidence association, new-track probation and bounded lost-track recovery while excluding donor detector/Torch, Kalman/SciPy/LAP/`cython_bbox`, OpenCV/native extensions, models, weights, ReID and biometric identity behavior.
- PR #61: bounded platform-neutral tracking evaluator for matched observations, misses, false-track observations, session-local ID switches, fragmentations, mean matched IoU and continuity. Synthetic fixtures are regression evidence only.
- PR #62: fail-closed UVify/NCSOFT 12-column tracking annotation parser pinned to `uvify-public/human_tracking_dataset@eb3af0cfe49de018a0c4736581daadd8eb860883`. Only source tracking IDs are mapped into evaluation identifiers; source person identity fields are not propagated.
- PR #63: cryptographic tracking-evidence admission boundary requiring exhaustive multi-object annotation scope, annotation SHA-256, ordered frame-manifest SHA-256, frame count and image dimensions before multi-object metrics are accepted.
- PR #64: evidence-byte rights admission requiring an authoritative rights source and explicit confirmation that commercial product-development evaluation is authorized. BDD100K was rejected for this path because its dataset terms do not establish general commercial rights for Analytics Lab.

## Current acceptance item — admit one exact rights-clear real sequence
The prior path was blocked by inaccessible UVify payload bytes and unsuitable fallbacks. The approach has therefore changed, as required by the efficiency policy, from scanning more prepackaged MOT datasets to admitting one tiny authoritative real sequence with clear commercial-evaluation rights and creating independent exhaustive labels for a fixed window.

Selected source: Wikimedia Commons `Video Codec Test pedestrian area 1080p25.y4m.webm`.

Reviewed authoritative facts from the pinned Commons file page:
- author: Taurus Media Technik;
- license: CC0 1.0 Universal Public Domain Dedication (`CC0-1.0`);
- commercial copying/modification/distribution/use is permitted under the reviewed dedication;
- canonical asset size: 11,215,394 bytes;
- published SHA-1: `51e89a672896e45cca17aa46cd223630a6266e26`;
- duration: 15.125 seconds;
- dimensions: 1920 x 1080;
- static pedestrian-area view with people passing close to the camera and high depth of field.

Rights/provenance review is recorded in `docs/dataset-review-wikimedia-cc0-tracking.md`. The exact rights source is pinned to the Commons file-page revision `oldid=1196486240`; the canonical media URL is pinned in `analytics_lab.wikimedia_tracking_admission`.

### First evidence head — digest discovery only
Branch `evidence/tracking-cc0-admission` adds a narrowly bounded hosted-runner probe that:
- streams only the exact canonical Wikimedia media URL;
- requires HTTPS and rejects a redirect outside the reviewed host/path;
- enforces the published `Content-Length` when present;
- caps the stream at 12,000,000 bytes;
- requires exact final size 11,215,394 bytes;
- requires exact published SHA-1 `51e89a672896e45cca17aa46cd223630a6266e26`;
- computes SHA-256 for the next fail-closed pin;
- retains/uploads no media, emits no frames, installs no detector/runtime, and makes no accuracy claim.

The evidence workflow is adjusted only for `evidence/tracking-cc0-*`; that branch runs this digest probe instead of the older generic GMDCSA/OMZ real-video lane. Normal branches and main do not acquire the asset. `AGENTS.md` is tightened to authorize exactly this one CC0 asset and exactly this digest-discovery behavior.

Focused local regression for the new admission code passed 5/5 before PR creation. The network evidence itself is deliberately unrun until the exact PR head executes on the GitHub-hosted Linux runner.

### Next measured comparison
If the first exact head verifies the published bytes and discovers SHA-256, use at most one remaining CI-triggering update in this session to pin that SHA-256 fail-closed and prepare a small fixed frame window selected before benchmark-model inspection. Do not claim tracking accuracy from digest discovery.

After the exact asset identity is pinned:
1. extract only the bounded fixed window ephemerally;
2. create independent exhaustive person boxes/session-local ground-truth IDs for every measured frame without using the benchmarked detector/tracker outputs as ground truth;
3. bind annotation SHA-256 and the ordered per-frame digest manifest through `TrackingEvidenceManifest`;
4. run the unchanged simple-IoU baseline and portable ByteTrack slice on identical detector observations;
5. compare raw fragmentation, ID switches, continuity, misses, false tracks, matched IoU, throughput/latency and CPU/resource cost;
6. attack only the largest demonstrated error source.

Do not admit Kalman/LAP/native extensions, another tracker donor, training, or parameter tuning until this comparison demonstrates a concrete need.

## Rejected/blocked tracking evidence sources retained for provenance
### UVify/NCSOFT
Repository `uvify-public/human_tracking_dataset`, pinned revision `eb3af0cfe49de018a0c4736581daadd8eb860883`, repository license blob `fab36c2f10dc9d2602bef9c9570df9c978598f03`, published CC-BY-4.0. The README describes 500 drone videos, 18,000 extracted images and multi-object/occlusion annotations, but the publisher's actual SharePoint payload remains unavailable through the bounded path. No sequence bytes or real tracking metric were admitted.

### D-PTUAC
Figshare DOI `10.6084/m9.figshare.24590568.v2`, CC-BY-4.0, is single-target visual-object tracking rather than exhaustive multi-object labeling. Its roughly 15 GB package is intentionally not downloaded because its N x 4 selected-target trajectories cannot support all-person false-track, ID-switch or fragmentation scoring.

### BDD100K
Repository `bdd100k/bdd100k` reviewed at `9ac17c6c7c51d2fc83065fccd707cd5b1882a293`. It is technically suitable for MOT, but its data/label terms do not establish general commercial product-development use for Analytics Lab outside qualifying BDD/BAIR Commons affiliation or a separate commercial license. No dataset media was downloaded.

## Retained person-down path — required secondary analytic
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. The detector/temporal path remains stable; repeated misses localize primarily to pose representation/association/posture continuity. OMZ `human-pose-estimation-0005` and `0006` were measured and rejected. Do not lower keypoint confidence, bridge genuine upright/other evidence, weaken 3000 ms persistence, extend the 750 ms gap ceiling, or shop another person detector merely to make holdouts pass.

Accepted evidence retained from prior work includes one legitimate GMDCSA Subject 1 positive emission with hard negatives clean, protected GMDCSA Subjects 2-4 as held-out failures/negatives, and five rights-reviewed Figshare activity pairs used for stage-attribution/generalization. Figshare activity labels do not provide defensible frame-level event intervals, so they are not scored for match/miss or alert latency and are never presented as commercial accuracy.

## Pose adaptation readiness — secondary and blocked
Selected trainer lineage remains `Daniil-Osokin/lightweight-human-pose-estimation.pytorch@d23c284b09acf27a163e1febd511e7482cac25ed`, Apache-2.0. Training remains blocked pending complete transitive dependency rights, authoritative checkpoint identity and artifact-specific commercial/redistribution rights, a rights-cleared subject-separated adaptation corpus, and an ONNX -> OpenVINO Runtime 2026.3.1 CPU export smoke test. GMDCSA Subjects 2-4 and all measured Figshare evidence subjects remain holdouts. Training is not on the North Star critical path.

## Runtime/model provenance
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Retained pose model `human-pose-estimation-0001` FP16.
- Retained evidence detector `person-detection-0200` FP16.
- OpenVINO Runtime `2026.3.1`.
- Evidence decoder `opencv-python-headless==4.12.0.88`.
- Portable ByteTrack association donor revision `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, MIT, adapted code only with no donor model/weights/data or new runtime dependency.

## Evidence/data provenance retained
### GMDCSA-24
Repository `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`, pinned revision `5abac7693229900cf80f722e878fbb119211fc1c`, reviewed repository license MIT, Zenodo DOI `10.5281/zenodo.13354453`, paper DOI `10.1016/j.dib.2024.110892`.

### Figshare 2017-activity source
Article `28596332`, version 2, reviewed license CC-BY-4.0, provenance `figshare:28596332:version-2`, activity mapping reference DOI `10.30970/eli.33.12` under CC-BY-4.0. Media remains outside public GitHub.

### Wikimedia CC0 pedestrian source
Pinned rights page `https://commons.wikimedia.org/w/index.php?title=File:Video_Codec_Test_pedestrian_area_1080p25.y4m.webm&oldid=1196486240`; canonical upload URL and published size/SHA-1 are pinned in repository code. SHA-256 is intentionally unknown until the first exact evidence run verifies the source bytes.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. This session started from live `main` `04b8c491432fb2850291d2d4ecf06470ce842ce1`, zero open implementation PRs, zero active runs for that head, zero unchanged retries, zero CI dispatches in this session and zero consecutive stalled sessions. Main run #157 was green on attempt 1.

Repository/head/open-PR/CI state and the unchanged efficiency-policy ceilings were verified before mutation. A local preflight reproduction using the pinned policy and current guardrail logic returned `{"allowed": true, "reasons": []}` for implementation. Focused local new-module regressions passed 5/5. Opening the PR below counts as CI-triggering action 1/2; no unchanged retry has been consumed.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Focused Detection + Tracking regressions:
```sh
python -m unittest tests.test_bytetrack tests.test_tracking_evaluation tests.test_uvify_tracking tests.test_tracking_evidence tests.test_wikimedia_tracking_admission -v
```

## Next executable decision
Open exactly one implementation PR from `evidence/tracking-cc0-admission` on unchanged base `04b8c491432fb2850291d2d4ecf06470ce842ce1`. Preserve the first-attempt outcome. Require the exact current head to pass Linux, Windows and the Analytics quality gate on the unchanged tested base before merge.

This first head is **digest discovery only**. It is not merge-eligible until the discovered SHA-256 is pinned in a changed second head and the same evidence probe passes fail-closed against that SHA-256, using at most the one remaining CI-triggering update this session. If the first run fails deterministically, diagnose/fix rather than rerunning unchanged.

## Outstanding commercial-release gates
For detection/tracking: substantially broader held-out real-video evidence across people/vehicles/objects, crowded scenes, crossings, occlusions, low light, viewpoints and resolutions; defensible detection precision/recall where labels permit; track fragmentation and ID-switch measurements; throughput/latency/resource envelope; Linux/Windows portability; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.

For later priorities, LPR/OCR, Face (including selectable face blurring), Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before any commercial claim.
