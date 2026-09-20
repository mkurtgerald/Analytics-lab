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

The remaining indispensable evidence is still an independently authored exhaustive person/track annotation package for frames 150-174, bound to those frame hashes, followed by one immutable detector-observation sequence fed identically to frozen simple-IoU and portable ByteTrack. Required untouched outputs remain ID switches, fragmentation, continuity, matched/missed observations, false-track observations, matched IoU, throughput/FPS, latency, elapsed time, CPU time and bounded resource cost. No tuning is permitted before measurement.

The deterministic blocker is unchanged: the approved hosted lane may hash but not export/label those decoded frames, while the current independent annotation environment cannot materialize the exact source pixels. No detector/tracker output may substitute for ground truth. Do not add tracker machinery until the exact independent-label path becomes executable.

## LPR/OCR — current critical path
PR #72 merged the first replaceable runnable engineering baseline into `main` at `2244bd2eee1f99fdfbc32f1ba636c288b7bb38a0`. PR #74 merged the first public-domain staged plate smoke at `e6d11cd3ea325b25c6fa9e5024d4936c6224ce5e`. PR #75 merged the first untouched real vehicle-scene detector diagnostic into `main` at `5edaef90083660a30667555e634639535f848040`; post-merge Analytics quality run #184 passed on attempt 1.

Current baseline:
- plate detector: Open Model Zoo `vehicle-license-plate-detection-barrier-0106` FP16 at exact OMZ commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0, exact size + SHA-384 verification before OpenVINO opens artifacts;
- runtime: reviewed OpenVINO `2026.3.1` CPU family;
- OCR engine: Tesseract `5.5.3` at exact commit `db0ec62f81b0737fbbe184d8fea40af5738f8eef`, Apache-2.0, optional local CLI;
- OCR model: `tesseract-ocr/tessdata_fast@87416418657359cb625c412a48b6e1d6d41c29bd`, `eng.traineddata` exactly 4,113,088 bytes with immutable Git blob `bbef4675053b5b468cdb477053e28b1c698ba08e`;
- data path: normalized plate box -> bounded in-memory BGR crop -> dependency-free RGB PPM -> Tesseract TSV -> normalized A-Z/0-9 observation;
- no automatic model/media download, image retention, identity behavior, training, or accuracy claim in the library.

### Preserved smoke results
The staged public-domain Wikimedia plate graphic `China license plate-Chongqing 渝A 92518.png` is pinned to SHA-256 `47ea02127b3c22856ac548164c822b104c74f7afb711a0cb43458080a11d5433`, with independently declared normalized text `A92518`. The untouched OMZ detector returned zero plate detections. Preserved first-attempt runtime was approximately 7.265 ms wall latency, 12.124 ms CPU and 137.64 one-image FPS.

PR #75 then measured the same untouched detector on the rights-cleared real Wikimedia photograph `PR China license plate Beijing 京B•K0074 Taxi.jpg`, pinned to SHA-1 `84ac1c7c66b10345fff12e7ed6876c46b91abbd3` and SHA-256 `f85e058f4526c43b34f97edf4349b8510b321db3fb98e5ce7a47b7f579e6f890`. The detector again returned zero plate detections. That source is a real vehicle scene, but it did not establish that the scene satisfied the detector's documented front-facing / minimum-plate-width operating envelope.

Both results are engineering-smoke evidence only. They are not roadway precision/recall, OCR accuracy, geography generalization, or commercial performance claims. No detector threshold or parameter was changed after either result.

### Current acceptance item — documented-envelope front-view smoke
Before spending a donor slot or tuning, run the frozen detector once on a source selected *before detector output* to satisfy its documented geometry: front-facing vehicle and a plate plainly wider than the published 96-pixel minimum.

Selected source:
- Wikimedia Commons `Land Rover Defender 110 (L316), front view.jpg`;
- rights page pinned to `oldid=1230217363`;
- author: Pittigrilli; source: own work;
- license: CC0 1.0 Universal Public Domain Dedication, including commercial reuse;
- dimensions: 4032x3024;
- published size: 2,680,061 bytes;
- published SHA-1: `8cdb3acc024e67267dabf6d3ac793d0c54924e85`;
- visible normalized plate text, independently declared before detector output: `MPR318`;
- detector-documented minimum plate width: 96 px;
- conservative human-authored plate-width lower bound before detector output: 1000 px. This is only a coarse envelope check, not a detector-derived box or accuracy annotation.

The first `evidence/lpr-front-envelope-*` head must be admission-only: verify the exact published size/SHA-1, discover and pin source SHA-256, emit no image, install no model/runtime, and perform no inference. Only the subsequent changed head may execute the same frozen detector and preserve its first result.

### Next executable LPR/OCR steps
1. Admit the selected CC0 front-view source and pin its immutable SHA-256 without inference.
2. On the same single implementation branch, run the untouched OMZ detector exactly once after the hash is pinned; preserve detections, best confidence, latency/FPS, elapsed time and CPU cost.
3. If it detects the clearly in-envelope plate, retain the detector provisionally and advance to the smallest rights-cleared independently labeled real plate/text set.
4. If it still misses, treat the result as measured justification for a bounded rights-clean detector replacement screen (maximum three candidates); do not tune blindly or add another framework.
5. Once a plate crop is available, exercise the existing Tesseract OCR contract and record raw normalized text, exact/character agreement where independently labeled, latency/FPS and CPU/resource cost.
6. Expand only to rights-cleared positives and negatives sufficient to measure plate misses/false positives and OCR exact/character reads. Tiny staged-real or single-image evidence must never be represented as commercial accuracy.

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

LPR/OCR has a runnable engineering baseline but still requires rights-cleared held-out real plate/text evidence, broader plate-style/geography validation, exact native-runtime/package dependency closure, notices/provenance review and release approval. The staged and single-scene smoke sources above do not reduce those commercial gates.

Face (including selectable face blurring), Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before any commercial claim.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
python -m unittest tests.test_lpr_ocr tests.test_lpr_wikimedia_evidence tests.test_lpr_vehicle_scene_evidence tests.test_lpr_front_envelope_evidence -v
```
