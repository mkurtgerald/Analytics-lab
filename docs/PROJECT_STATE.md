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
Live `main` at this work item's start is `853c426bbd4898db44407068fb58248e7dde0531`, the merge of PR #62. Post-merge Analytics quality run #153 passed on attempt 1. At start there were zero open implementation PRs and zero queued/running runs for that exact head.

Landed boundaries:
- PR #59: detector-neutral normalized detection/tracked-object contract.
- PR #60: portable ByteTrack-derived association slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, MIT. It keeps the defining high-confidence then low-confidence association, new-track probation and bounded lost-track recovery while excluding donor detector/Torch, Kalman/SciPy/LAP/`cython_bbox`, OpenCV/native extensions, models, weights, ReID and biometric identity behavior.
- PR #61: bounded platform-neutral tracking evaluator for matched observations, misses, false-track observations, session-local ID switches, fragmentations, mean matched IoU and continuity. Synthetic fixtures are regression evidence only.
- PR #62: fail-closed UVify/NCSOFT 12-column tracking annotation parser pinned to `uvify-public/human_tracking_dataset@eb3af0cfe49de018a0c4736581daadd8eb860883`. Only source tracking IDs are mapped into evaluation identifiers; source person identity fields are not propagated.

### Current acceptance item — cryptographically bound multi-object evidence
The evaluator's false-track, ID-switch, fragmentation and continuity outputs are only defensible when labels are exhaustive for all measured objects. A crowded clip with a single annotated target does not satisfy that condition.

`analytics_lab.tracking_evidence` therefore fails closed before multi-object scoring unless one exact sequence records: dataset/provenance/version/license/attribution, sequence ID, `annotation_scope=exhaustive_multi_object`, annotation SHA-256, a deterministic SHA-256 over the ordered per-frame digest manifest, positive frame count, and bounded image dimensions. Media stays ephemeral/outside GitHub. Synthetic manifest tests validate only this contract.

The primary evidence source remains UVify/NCSOFT. Its pinned README describes 500 drone videos, 18,000 extracted images, multi-object human tracking annotations and 49,258 occluded object annotations. The repository license is CC-BY-4.0 at blob `fab36c2f10dc9d2602bef9c9570df9c978598f03`. The publisher's actual dataset payload is SharePoint-hosted and remains unavailable through the current bounded execution path, so no sequence payload, media hash or real tracking metric is claimed yet.

The previously identified D-PTUAC Figshare fallback is now rejected for this acceptance item. Its paper describes 138 sequences / more than 121k frames under CC-BY-4.0, but the dataset is visual-object tracking of a selected target: each sequence's `groundtruth.txt` is an N x 4 target box `[xmin, ymin, width, height]`. Crowds in the video therefore do not make its annotation scope exhaustive multi-object ground truth. Pulling its roughly 15.01 GB package would not support the evaluator's false-track or multi-ID metrics and is intentionally avoided.

### Next measured comparison
Obtain the smallest exact rights-cleared UVify/NCSOFT test sequence if a bounded publisher payload path becomes available. Record exact source revision, attribution, annotation digest, ordered frame-manifest digest, frame count and dimensions, keep bytes ephemeral, then run the unchanged simple-IoU association baseline and portable ByteTrack slice on that same sequence. Compare fragmentation, ID switches, continuity, misses, false tracks, matched IoU, throughput/latency and resource cost. Do not admit heavier Kalman/LAP/native-extension machinery unless those measurements localize the largest remaining error to motion prediction/global assignment.

If UVify remains inaccessible, review at most one additional source for exhaustive multi-object labels and explicit commercial-compatible data rights before changing approach. Do not use single-target, partial-actor, noncommercial, research-only or unknown-rights data for the real tracking acceptance claim.

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

### Rejected tracking evidence source for this metric protocol
D-PTUAC, Figshare DOI `10.6084/m9.figshare.24590568.v2`, published CC-BY-4.0, is single-target visual-object tracking rather than exhaustive multi-object labeling. Do not use it to score false tracks, multi-object ID switches or fragmentation in this evaluator.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. This work item began from live `main` `853c426bbd4898db44407068fb58248e7dde0531`, zero open implementation PRs, zero active runs for that head, zero unchanged retries, zero CI dispatches in this session and zero consecutive stalled sessions. Main run #153 was green on attempt 1.

Repository/head/open-PR/CI state and the unchanged efficiency-policy ceilings were verified before mutation. The preflight snapshot reproduced locally from the pinned guardrail/policy returned `{"allowed": true, "reasons": []}`. The change adds no donor, dependency, model, weight, media, identity behavior, threshold change, paid resource or training.

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
Open exactly one implementation PR from `feature/tracking-evidence-admission` on unchanged base `853c426bbd4898db44407068fb58248e7dde0531`. Require the exact head to pass Linux, Windows and the Analytics quality gate; merge only if the tested head/base remain unchanged and all required checks are green.

After merge, return to real labeled evidence acquisition. Do not implement another tracking donor before a qualifying sequence is admitted and the existing simple-IoU and portable ByteTrack paths are measured on identical evidence.

## Outstanding commercial-release gates
For detection/tracking: substantially broader held-out real-video evidence across people/vehicles/objects, crowded scenes, crossings, occlusions, low light, viewpoints and resolutions; defensible detection precision/recall where labels permit; track fragmentation and ID-switch measurements; throughput/latency/resource envelope; Linux/Windows portability; privacy/security/provenance review; dependency/notices review; versioned installable integration adapter; packaging; and explicit owner commercial-release approval.

For later priorities, LPR/OCR, Face (including selectable face blurring), Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before any commercial claim.
