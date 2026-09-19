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
Live `main` at this work item's start is `b5f05f57b7a2677be1d9c7c5bed547784187db3b`, the merge of PR #63. Post-merge Analytics quality run #155 passed on attempt 1. At start there were zero open implementation PRs and zero queued/running runs for that exact head.

Landed boundaries:
- PR #59: detector-neutral normalized detection/tracked-object contract.
- PR #60: portable ByteTrack-derived association slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, MIT. It keeps the defining high-confidence then low-confidence association, new-track probation and bounded lost-track recovery while excluding donor detector/Torch, Kalman/SciPy/LAP/`cython_bbox`, OpenCV/native extensions, models, weights, ReID and biometric identity behavior.
- PR #61: bounded platform-neutral tracking evaluator for matched observations, misses, false-track observations, session-local ID switches, fragmentations, mean matched IoU and continuity. Synthetic fixtures are regression evidence only.
- PR #62: fail-closed UVify/NCSOFT 12-column tracking annotation parser pinned to `uvify-public/human_tracking_dataset@eb3af0cfe49de018a0c4736581daadd8eb860883`. Only source tracking IDs are mapped into evaluation identifiers; source person identity fields are not propagated.
- PR #63: cryptographic tracking-evidence admission boundary requiring exhaustive multi-object annotation scope, annotation SHA-256, ordered frame-manifest SHA-256, frame count and image dimensions before multi-object metrics are accepted.

### Current acceptance item — bind evidence rights as well as bytes
The evaluator's false-track, ID-switch, fragmentation and continuity outputs are only defensible when labels are exhaustive for all measured objects and the exact evidence bytes are authorized for commercial product-development evaluation.

`analytics_lab.tracking_evidence` therefore fails closed unless one exact sequence records dataset/provenance/version/license/attribution, an authoritative rights source, explicit confirmation that commercial evaluation of the evidence bytes is authorized, `annotation_scope=exhaustive_multi_object`, annotation SHA-256, a deterministic SHA-256 over the ordered per-frame digest manifest, positive frame count, and bounded image dimensions. Media stays ephemeral/outside GitHub. Synthetic manifest tests validate only this contract.

This closes a demonstrated provenance gap: a permissive code-repository license or public download URL does not necessarily grant commercial rights to the underlying dataset media.

The primary evidence source remains UVify/NCSOFT. Its pinned README describes 500 drone videos, 18,000 extracted images, multi-object human tracking annotations and 49,258 occluded object annotations. The repository license is CC-BY-4.0 at blob `fab36c2f10dc9d2602bef9c9570df9c978598f03`. The publisher's actual dataset payload is SharePoint-hosted and remains unavailable through the current bounded execution path, so no sequence payload, media hash or real tracking metric is claimed yet.

### Additional source review — BDD100K rejected for this commercial path
Per the bounded fallback rule, exactly one additional technically relevant source was reviewed: `bdd100k/bdd100k@9ac17c6c7c51d2fc83065fccd707cd5b1882a293`.

BDD100K supports multi-object detection tracking, but its authoritative `doc/source/license.rst` separates the BSD-3-Clause code/resources license from the downloaded data/label terms. The data/label grant permits educational, research and not-for-profit use generally, while commercial use is granted to BDD and BAIR Commons members and their affiliates and otherwise points users to UC Berkeley for commercial licensing opportunities. No qualifying membership or separate commercial license is established for Analytics Lab, so BDD100K is rejected for the current commercial acceptance path. No BDD100K media or bulk archive was downloaded. See `docs/dataset-review-bdd100k-tracking.md`.

The previously identified D-PTUAC Figshare fallback remains rejected for this acceptance item. Its paper describes 138 sequences / more than 121k frames under CC-BY-4.0, but the dataset is visual-object tracking of a selected target: each sequence's `groundtruth.txt` is an N x 4 target box `[xmin, ymin, width, height]`. Crowds in the video therefore do not make its annotation scope exhaustive multi-object ground truth. Pulling its roughly 15.01 GB package would not support the evaluator's false-track or multi-ID metrics and is intentionally avoided.

### Next measured comparison
Because UVify remains inaccessible and the one permitted additional prepackaged source review did not clear commercial rights, change approach rather than scanning more MOT datasets.

Use the smallest rights-cleared real video sequence from an authoritative permissive source (for example, CC0/CC-BY source media with terms covering commercial analysis) and create a tiny in-house exhaustive multi-object annotation set for that sequence, independent of model output. Record the exact source revision/URL, license/attribution, authoritative rights source, annotation digest, ordered frame-manifest digest, frame count and dimensions; keep media bytes ephemeral. Then run the unchanged simple-IoU association baseline and portable ByteTrack slice on identical evidence and compare fragmentation, ID switches, continuity, misses, false tracks, matched IoU, throughput/latency and resource cost.

Do not search another prepackaged MOT dataset in this path unless the approach changes again after measured failure. Do not admit heavier Kalman/LAP/native-extension machinery unless the measured comparison localizes the largest remaining error to motion prediction/global assignment.

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

## Evidence/data provenance
### GMDCSA-24
Repository `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`, pinned revision `5abac7693229900cf80f722e878fbb119211fc1c`, reviewed repository license MIT, Zenodo DOI `10.5281/zenodo.13354453`, paper DOI `10.1016/j.dib.2024.110892`.

### Figshare 2017-activity source
Article `28596332`, version 2, reviewed license CC-BY-4.0, provenance `figshare:28596332:version-2`, activity mapping reference DOI `10.30970/eli.33.12` under CC-BY-4.0. Media remains outside public GitHub.

### UVify/NCSOFT human tracking source candidate
Repository `uvify-public/human_tracking_dataset`, pinned revision `eb3af0cfe49de018a0c4736581daadd8eb860883`, repository license blob `fab36c2f10dc9d2602bef9c9570df9c978598f03`, published CC-BY-4.0. Exact data payload is not admitted or hashed.

### Rejected tracking evidence sources for the current metric/commercial protocol
- D-PTUAC, Figshare DOI `10.6084/m9.figshare.24590568.v2`, published CC-BY-4.0, is single-target visual-object tracking rather than exhaustive multi-object labeling. Do not use it to score false tracks, multi-object ID switches or fragmentation in this evaluator.
- BDD100K repository `bdd100k/bdd100k`, reviewed at `9ac17c6c7c51d2fc83065fccd707cd5b1882a293`, is technically suitable for MOT but its downloaded data/label terms do not establish general commercial product-development rights. Do not use it for this commercial acceptance path without separately established rights.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. This work item began from live `main` `b5f05f57b7a2677be1d9c7c5bed547784187db3b`, zero open implementation PRs, zero active runs for that head, zero unchanged retries, zero CI dispatches in this session and zero consecutive stalled sessions. Main run #155 was green on attempt 1.

Repository/head/open-PR/CI state and the unchanged efficiency-policy ceilings were verified before mutation. The preflight snapshot reproduced locally from the pinned guardrail/policy returned `{"allowed": true, "reasons": []}`. The change adds no donor runtime, dependency, model, weight, media, identity behavior, threshold change, paid resource or training.

Focused local regression for the changed tracking-evidence boundary passed 7/7 before push. Full Linux/Windows/quality-gate verification remains required on the exact PR head before merge.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Focused Detection + Tracking regressions:
```sh
python -m unittest tests.test_bytetrack tests.test_tracking_evaluation tests.test_uvify_tracking tests.test_tracking_evidence -v
```

## Next executable decision
Open exactly one implementation PR from `feature/tracking-rights-admission` on unchanged base `b5f05f57b7a2677be1d9c7c5bed547784187db3b`. Require the exact head to pass Linux, Windows and the Analytics quality gate; merge only if the tested head/base remain unchanged and all required checks are green.

After merge, move directly to the changed evidence-acquisition approach: one tiny authoritative CC0/CC-BY real sequence plus independent exhaustive in-house annotations, cryptographically bound before measurement. Do not start another tracker implementation or scan another prepackaged MOT dataset first.

## Outstanding commercial-release gates
For detection/tracking: substantially broader held-out real-video evidence across people/vehicles/objects, crowded scenes, crossings, occlusions, low light, viewpoints and resolutions; defensible detection precision/recall where labels permit; track fragmentation and ID-switch measurements; throughput/latency/resource envelope; Linux/Windows portability; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.

For later priorities, LPR/OCR, Face (including selectable face blurring), Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before any commercial claim.
