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

## Detection + tracking — frozen at first real-video acceptance gate
Tracking remains stopped at the policy's three-session threshold. The implementation boundary is intact: detector-neutral tracking contract, portable ByteTrack slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` (MIT), bounded evaluator, rights/evidence admission, pre-registered Wikimedia CC0 frames 150-174, canonical RGB24 frame identities, exhaustive ground-truth package binding, frozen simple-IoU control, and first-attempt head-to-head runner.

The fixed frame-manifest SHA-256 is `52f00a4fc1e013d9c0cae9647cb386369441f005a970d6d42c449cbd0269a031`; benchmark-plan SHA-256 is `eb7995a389a22f3528b9bfca97d64f9d8e3c515ebf1c764cf25b68fbdcc3479c`.

The remaining indispensable evidence is an independently authored exhaustive person/track annotation package for frames 150-174, bound to those hashes, followed by one immutable detector-observation sequence fed identically to frozen simple-IoU and portable ByteTrack. Required untouched outputs remain ID switches, fragmentation, continuity, matched/missed observations, false-track observations, matched IoU, throughput/FPS, latency, elapsed time, CPU time and bounded resource cost. No tuning is permitted before measurement.

The deterministic blocker is unchanged: the approved hosted lane may hash but not export/label those decoded frames, while the current independent annotation environment cannot materialize the exact source pixels. No detector/tracker output may substitute for ground truth. Do not add tracker machinery until the exact independent-label path becomes executable.

## LPR/OCR — current critical path
PR #72 merged the first replaceable runnable baseline. PR #74 preserved the first staged public-domain plate smoke. PR #75 merged the first untouched real vehicle-scene detector diagnostic at `5edaef90083660a30667555e634639535f848040`; post-merge Analytics quality run #184 passed on attempt 1.

Current baseline:
- plate detector: Open Model Zoo `vehicle-license-plate-detection-barrier-0106` FP16 at exact OMZ commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0, exact artifact size + SHA-384 verified before OpenVINO opens it;
- runtime: reviewed OpenVINO `2026.3.1` CPU family;
- OCR engine: Tesseract `5.5.3` at exact commit `db0ec62f81b0737fbbe184d8fea40af5738f8eef`, Apache-2.0;
- OCR model: `tesseract-ocr/tessdata_fast@87416418657359cb625c412a48b6e1d6d41c29bd`, `eng.traineddata` 4,113,088 bytes, immutable Git blob `bbef4675053b5b468cdb477053e28b1c698ba08e`;
- data path: normalized plate box -> bounded in-memory BGR crop -> dependency-free RGB PPM -> Tesseract TSV -> normalized A-Z/0-9 observation;
- no automatic model/media download, image retention, identity behavior, training, or accuracy claim in the library.

### Preserved smoke results
The staged Wikimedia plate graphic `China license plate-Chongqing 渝A 92518.png`, source SHA-256 `47ea02127b3c22856ac548164c822b104c74f7afb711a0cb43458080a11d5433`, produced zero plate detections from the untouched OMZ detector. Preserved first-attempt runtime was approximately 7.265 ms wall latency, 12.124 ms CPU and 137.64 one-image FPS. This is staged-image smoke evidence only.

PR #75 then measured the same untouched detector on the rights-cleared real Wikimedia photograph `PR China license plate Beijing 京B•K0074 Taxi.jpg`, source SHA-256 `f85e058f4526c43b34f97edf4349b8510b321db3fb98e5ce7a47b7f579e6f890`. It again returned zero plate detections. That scene did not prove it met the detector's documented front-facing / >=96-pixel-plate envelope, so it was not sufficient by itself to reject the detector.

Neither result is roadway precision/recall, OCR accuracy, geography generalization, or commercial performance evidence. No detector threshold or parameter changed after either result.

### Current acceptance item — documented-envelope front-view smoke
PR #76 uses the single implementation lane to make the smallest results-first decision before tuning or screening replacement donors. The frozen detector is being tested on a source selected *before detector output* to meet its documented geometry: a front-facing vehicle whose plate is plainly wider than the published 96-pixel minimum.

Selected source:
- Wikimedia Commons `Land Rover Defender 110 (L316), front view.jpg`;
- rights page pinned to `oldid=1230217363`;
- author Pittigrilli, own work, CC0-1.0 including commercial reuse;
- dimensions 4032x3024;
- published size 2,680,061 bytes;
- published SHA-1 `8cdb3acc024e67267dabf6d3ac793d0c54924e85`;
- immutable SHA-256 discovered on untouched admission-only PR #76 head: `32e5637e39b54c26192c011c1cc5516bd6d35573582b8e09d7bc9aae90ef1db4`;
- visible normalized plate text independently declared before detector output: `MPR318`;
- detector-documented minimum plate width: 96 px;
- conservative human-authored premeasurement plate-width lower bound: 1000 px. This is an envelope check only, not a detector-derived box or accuracy annotation.

PR #76 admission-only run #185 passed Linux, Windows, the dedicated bounded CC0 evidence step and the Analytics quality gate on attempt 1 at head `fe6e179ea591e89c1e5bba489a41ae36fe1787c4`. The evidence step verified the exact published source identity, discovered the SHA-256 above, reported the source within the preregistered plate-size envelope, and performed no model/runtime install or inference.

The SHA-256 is now pinned fail-closed on the same branch. The next exact-head run is authorized to install only the already-reviewed `openvino==2026.3.1` and `opencv-python-headless==4.12.0.88`, download only the already-reviewed hash-pinned OMZ detector artifacts, execute the untouched detector once, and emit only aggregate detections/confidence/runtime evidence. No tuning is permitted before this first measurement.

### Decision after this measurement
- If the frozen detector detects the clearly in-envelope plate, retain it provisionally and advance to the smallest rights-cleared independently labeled real plate/text set, then exercise the existing Tesseract path.
- If it still misses, treat that first untouched in-envelope result as measured justification for a bounded rights-clean detector replacement screen of at most three candidates. Do not tune blindly or add another framework.
- A single image remains engineering-smoke evidence only and must never be represented as commercial accuracy.

## Retained person-down / slip-fall path
The required secondary path remains:
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`.

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. Existing detector/temporal behavior and hard negatives remain protected while higher-priority shared perception/evaluation work advances.

## Outstanding commercial-release gates
Detection/tracking still needs broader held-out real-video evidence across people/vehicles/objects, crowded scenes, crossings, occlusions, low light, viewpoints and resolutions; defensible precision/recall where labels permit; track fragmentation/ID-switch measurements; throughput/latency/resource envelopes; privacy/security/provenance; versioned integration; packaging; and explicit owner commercial-release approval.

LPR/OCR still needs rights-cleared held-out real plate/text positives and negatives, broader plate-style/geography validation, exact native-runtime/package dependency closure, notices/provenance review, and explicit release approval. The smoke sources above do not reduce those commercial gates.

Face (including selectable face blurring), Weapons and Appearance Search each require their own rights-cleared donor/model/data review and held-out validation before commercial claims.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
python -m unittest tests.test_lpr_ocr tests.test_lpr_wikimedia_evidence tests.test_lpr_vehicle_scene_evidence tests.test_lpr_front_envelope_evidence -v
```
