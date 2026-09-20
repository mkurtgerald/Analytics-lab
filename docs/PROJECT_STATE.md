# Project state — 2026-09-20

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

### Permanent priority
1. Detection + tracking.
2. LPR/OCR.
3. Face — including required selectable face blurring with policy-driven default blur, permission-gated unblur/reblur, server-side enforcement, auditability and preservation of original evidence.
4. Weapons.
5. Appearance search.

Person-down and slip/fall remain required deliverables and are secondary only in sequencing.

## Detection + tracking — frozen at the first real-video acceptance gate
Tracking remains stopped at the policy's three-session threshold. The implementation boundary is intact: detector-neutral tracking contract, portable ByteTrack slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` (MIT), bounded evaluator, rights/evidence admission, pre-registered Wikimedia CC0 frames 150-174, canonical RGB24 frame identities, exhaustive ground-truth package binding, frozen simple-IoU control, and first-attempt head-to-head runner.

The fixed frame-manifest SHA-256 is `52f00a4fc1e013d9c0cae9647cb386369441f005a970d6d42c449cbd0269a031`; the benchmark-plan SHA-256 is `eb7995a389a22f3528b9bfca97d64f9d8e3c515ebf1c764cf25b68fbdcc3479c`.

The remaining indispensable tracking evidence is still an independently authored exhaustive person/track annotation package for frames 150-174, bound to those frame hashes, followed by one immutable detector-observation sequence fed identically to frozen simple-IoU and portable ByteTrack. Required untouched first-attempt outputs remain ID switches, fragmentation, continuity, matched/missed observations, false-track observations, matched IoU, throughput/FPS, latency, elapsed time, CPU time and bounded resource cost. No tuning is permitted before measurement.

The deterministic blocker is unchanged: the approved hosted lane may hash but not export/label those decoded frames, while the current independent annotation environment cannot materialize the exact source pixels. No detector/tracker output may substitute for ground truth. Do not add tracker machinery until the exact independent-label path becomes executable.

## LPR/OCR — current critical path
PR #72 merged the first replaceable runnable engineering baseline into `main` at `2244bd2eee1f99fdfbc32f1ba636c288b7bb38a0`. PR run #173 and post-merge run #174 both passed Linux, Windows and the Analytics quality gate on attempt 1.

Current baseline:
- plate detector: Open Model Zoo `vehicle-license-plate-detection-barrier-0106` FP16 at exact OMZ commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0, exact size + SHA-384 verification before OpenVINO opens artifacts;
- runtime: reviewed OpenVINO `2026.3.1` CPU family;
- OCR engine: Tesseract `5.5.3` at exact commit `db0ec62f81b0737fbbe184d8fea40af5738f8eef`, Apache-2.0, optional local CLI;
- OCR model: `tesseract-ocr/tessdata_fast@87416418657359cb625c412a48b6e1d6d41c29bd`, `eng.traineddata` exactly 4,113,088 bytes with immutable Git blob `bbef4675053b5b468cdb477053e28b1c698ba08e`;
- data path: normalized plate box -> bounded in-memory BGR crop -> dependency-free RGB PPM -> Tesseract TSV -> normalized A-Z/0-9 observation;
- no automatic model/media download, image retention, identity behavior, training, or accuracy claim in the library.

### Current acceptance item — rights-cleared plate/text smoke evidence
The next item is the smallest evidence package that can exercise the baseline without customer/home media or pretending a staged sample establishes product accuracy.

Selected smoke source:
- Wikimedia Commons `China license plate-Chongqing 渝A 92518.png`;
- rights page pinned to `oldid=856647637`;
- author: Sdee at Chinese Wikipedia;
- rights: author-released public domain / permission for use for any purpose;
- canonical upload host: `upload.wikimedia.org`;
- published dimensions: 680x144;
- published byte size: 61,465;
- published SHA-1: `d365a117631a8fa5a2a0fb7a8d2a03fe2e9b73bc`;
- independently declared normalized alphanumeric plate text for this engineering smoke: `A92518` (the Chinese province glyph is deliberately outside the current A-Z/0-9 baseline contract).

`analytics_lab.lpr_wikimedia_evidence` adds a bounded fail-closed admission path for exactly this source. The first evidence head intentionally leaves SHA-256 unpinned: it may only verify the reviewed host, byte cap, published size/SHA-1 and emit the discovered SHA-256 if explicitly run. Inference is forbidden until a changed second head pins that SHA-256. The same module already contains the narrow post-pin CPU detector measurement path so no second framework is needed: exact OMZ artifacts are size/SHA-384 checked, pixels stay in memory, temporary model files are deleted, and aggregate detector confidence/IoU plus wall/CPU/FPS/latency are emitted.

This source is a staged/sample plate graphic, not a roadway/camera sample. It may prove only that the evidence and detector path execute deterministically. It must never be cited as commercial LPR accuracy or real-world generalization.

Focused local pure-boundary regression for the new admission contract passed 3/3 before repository mutation. Repository CI must still run before merge; synthetic/unit fixtures are not accuracy evidence.

### Next executable LPR/OCR steps
1. Merge the evidence-admission boundary only after exact-head Linux, Windows and Analytics quality gate are green on unchanged base `2244bd2eee1f99fdfbc32f1ba636c288b7bb38a0`.
2. Add/execute a narrowly scoped hosted LPR evidence lane only if it preserves AGENTS.md ceilings and existing five-minute/read-only/no-artifact-upload controls; use it to discover and pin the source SHA-256, then run the untouched detector smoke on the same source.
3. Close the remaining OCR runtime portability gap without weakening the exact Tesseract/version/artifact contract. Do not introduce a second OCR framework unless a measured runtime or recognition requirement justifies it.
4. Expand only after the smoke path works to the smallest additional rights-cleared, independently labeled real plate/text set with positives and negatives; measure plate misses/false positives, exact/character reads, latency/FPS and CPU/resource cost.

## Retained person-down / slip-fall path
The required secondary path remains:
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`.

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. Existing detector/temporal behavior and hard negatives remain protected while higher-priority shared perception/evaluation work advances.

## Runtime/model provenance retained
- Open Model Zoo commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- OpenVINO Runtime `2026.3.1`.
- Retained person-down detector `person-detection-0200` FP16 and pose model `human-pose-estimation-0001` FP16.
- Portable ByteTrack donor `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, MIT, association slice only.
- Tesseract `5.5.3` commit `db0ec62f81b0737fbbe184d8fea40af5738f8eef`, Apache-2.0.
- tessdata_fast English model commit `87416418657359cb625c412a48b6e1d6d41c29bd`, Apache-2.0.

## Outstanding commercial-release gates
Detection/tracking still requires broader held-out real-video evidence across people/vehicles/objects, crowded scenes, crossings, occlusions, low light, viewpoints and resolutions; defensible precision/recall where labels permit; track fragmentation/ID-switch measurements; throughput/latency/resource envelopes; privacy/security/provenance; versioned integration; packaging; and explicit owner commercial-release approval.

LPR/OCR has a runnable engineering baseline but still requires rights-cleared held-out real plate/text evidence, broader plate-style/geography validation, exact native-runtime/package dependency closure, notices/provenance review and release approval. The staged public-domain smoke source above does not reduce those commercial gates.

Face (including selectable face blurring), Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before any commercial claim.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
python -m unittest tests.test_lpr_ocr tests.test_lpr_wikimedia_evidence -v
```
