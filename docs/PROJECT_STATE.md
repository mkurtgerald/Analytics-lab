# Project state — 2026-09-24

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

### Permanent priority
1. Detection + tracking.
2. LPR/OCR.
3. Face — including required selectable face blurring with policy-driven default blur, permission-gated unblur/reblur, server-side enforcement, auditability and preservation of original evidence.
4. Weapons.
5. Appearance search.

Person-down and slip/fall remain required deliverables and are secondary only in sequencing. On September 24, 2026 the owner-directed Face hardening finish line was reached; the LPR/OCR smallest runnable engineering baseline then closed in PR #111 and the single active implementation lane moved to Weapons after exact-head and post-merge verification. PR #84 remains closed/unmerged as preserved historical work, and the network-dependent glyph-evidence hypothesis remains closed under the three-session stop rule. Face feature expansion is stopped except for regression, packaging/provenance, and later explicitly authorized evidence needed to preserve the accepted boundary.

## Detection + tracking — frozen at first real-video acceptance gate
Tracking remains stopped at the policy's three-session threshold. The retained boundary includes the detector-neutral tracking contract, portable ByteTrack slice pinned to `FoundationVision/ByteTrack@d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` (MIT), bounded evaluator, rights/evidence admission, pre-registered Wikimedia CC0 frames 150-174, canonical RGB24 frame identities, exhaustive ground-truth package binding, frozen simple-IoU control, and first-attempt head-to-head runner.

The fixed frame-manifest SHA-256 is `52f00a4fc1e013d9c0cae9647cb386369441f005a970d6d42c449cbd0269a031`; benchmark-plan SHA-256 is `eb7995a389a22f3528b9bfca97d64f9d8e3c515ebf1c764cf25b68fbdcc3479c`.

The remaining blocker is unchanged: an independently authored exhaustive person/track annotation package for those exact frames cannot currently be materialized in the independent annotation environment without violating the approved evidence lane. No detector/tracker output may substitute for ground truth and no further tracker machinery should be added until that exact label path becomes executable.

## LPR/OCR — smallest runnable engineering baseline closed
The retained runnable baseline is Open Model Zoo `vehicle-license-plate-detection-barrier-0106` FP16 behind a replaceable detector boundary, OpenVINO `2026.3.1` CPU, Tesseract `5.5.3` with pinned `tessdata_fast` English data, and the model-free OpenCV 4.12.0 proposal fallback. This is an engineering baseline only; it is not a commercial-accuracy or release-readiness claim.

Exact rights-cleared CC0 OCR evidence is already preserved from successful hosted-Windows runs. The preregistered gray/Otsu Land Rover run #202 observed `MPR318 -> IFMPR318J`; South Carolina run #206 observed `354AVV -> 354AVIV`; Washington run #209 observed `CPU4704 -> CPU404`. The frozen aggregate Levenshtein edit distance is therefore `5` across three samples. These misses are retained as measured evidence and must not be normalized away or used for image-specific tuning.

The donor screen previously reached its three-candidate ceiling without admitting replacement OCR weights because training-data/weight commercial provenance did not close strongly enough. The project-authored OpenCV glyph fallback was then attempted as a no-weights alternative. Runs #211 and #212 exposed pre-measurement harness defects; the crop-interface mismatch was diagnosed. After Face completed, successor PR #111 resumed that hypothesis. Run #276 and changed-approach Linux run #277 both failed before any aggregate glyph comparison result because the pinned Wikimedia acquisition path returned HTTP 429. Synthetic glyph regressions passed, but no rights-cleared real glyph observation was produced. Under the three-session stop rule, this network-dependent glyph-evidence hypothesis is now closed without acceptance or rejection of the recognizer; do not retry it unchanged, do not tune it from synthetic fixtures, and do not substitute unpinned media.

The smallest runnable LPR/OCR engineering baseline is therefore frozen at the already-verified Tesseract 5.5.3 path plus the existing replaceable detector/proposal boundaries. Remaining non-baseline gates are broader independently labeled rights-cleared held-out evidence, a stronger OCR donor only if code/weight/data rights close cleanly, native dependency/redistribution notices, representative soak/performance work, and explicit commercial-release approval. PR #111 exact head passed run #278, merged to main as `2aa727be375f6894f6a709a6d29c65f21ad163bb`, and post-merge run #279 passed; the single active implementation lane is now Weapons. Tracking remains frozen at its independent-label blocker.

## Weapons — active donor-first engineering lane
The selected donor path reuses the already-admitted Google/TensorFlow standard Open Images V4 SSD MobileNetV2 artifact and OpenVINO `2026.3.1` CPU runtime used by Face. No second detector/runtime is introduced. Provenance remains pinned to `tensorflow/models@0558408514dacf2fe2860cd72ac56cbdf62a24c0` under Apache-2.0, with corrected OIDv4 label-map blob `643b9e8ed5d9239a3248b895fb32f3b51caa92f3`. The fixed allowlist is Knife 285, Kitchen knife 325, Rifle 351, Shotgun 361, Sword 365, Weapon 408, and Handgun 533.

The current acceptance-moving item is the smallest model-output integration boundary: reuse the admitted OIDv4 exact ports, normalized-box validation, fail-closed parsing, 300x300 preprocessing contract and one-compile reusable runtime lifecycle behind a K5-owned allowlist adapter that emits existing model-neutral `DetectionCandidate` values. Initial regression evidence is synthetic/model-output contract evidence only and must not be presented as detection accuracy, false-positive performance, field readiness, or commercial-release approval. No home/customer media, new model/data source, threshold tuning, paid resource, GPU job, recognition/identity feature, or second framework is authorized by this boundary.

A later rights-cleared evidence lane, if separately authorized in AGENTS.md, must independently verify media rights and preserve untouched detector observations before any engineering-quality or accuracy statement. Ultralytics remains inadmissible in the proprietary path without a separately authorized commercial license; RF-DETR remains only a reserve candidate because it would introduce a new fine-tuning/custom-class path without a measured need.

## Retained person-down / slip-fall path
The required secondary path remains:
`authorized local video -> person-detection-0200 -> bounded spatial continuity/orientation recovery -> OMZ human-pose-estimation-0001 -> corrected OpenPose decode -> bounded pose association -> conservative posture classification -> 3000 ms temporal persistence with at most 750 ms bounded unknown-gap tolerance -> evidence-linked candidate -> labeled evaluation`.

Candidate events never infer injury, cause, fault, intent, negligence or medical condition. Existing detector/temporal behavior and hard negatives remain protected while higher-priority shared perception/evaluation work advances.

## Face detection + automated/selectable blurring — ~90% ENGINEERING maturity; feature expansion stopped
The privacy/product boundary is already in place independently of model admission: normalized face boxes, immutable local-model verification, replaceable detector adapter, default-on Gaussian blur with bounded box expansion, permission-gated unblur/reblur, unauthorized-unblur denial, input-frame immutability, bounded face counts, and privacy audit state containing no biometric identity. Face recognition, embeddings, ReID and identity matching remain explicitly out of scope.

### September 24 engineering-maturity checkpoint
PR #110 hardened the reusable standard-OID/OpenVINO runtime so model conversion/CPU compilation occurs once per runtime instance and subsequent detections reuse the compiled model. Exact head `436854718f2fbc261bc74dc715ad74713473cb30` passed run #274 on unchanged base `d4a76770690171656a789c025ad88aab4511dd3a`; it merged to `main` as `8d42e7110419900a9e8d8ef9f97df80fb0774fdd`, and post-merge run #275 passed. The bounded hardening evidence used only deterministic synthetic input for three sequential calls and recorded compile count, per-call engineering timing, peak process RSS and input immutability. Existing regression coverage includes zero-face output, multiple class-502 filtering, non-face rejection, malformed-output fail-closed behavior, default blur, denied-unblur remains blurred, authorized-unblur passthrough, input immutability, max-face bounds, and bounded large-region Gaussian planning. The earlier admitted CC0 real portrait remains the sole real-person execution sample and is not a basis for an accuracy claim.

Face is therefore recorded as approximately **90% ENGINEERING maturity only**. This is not commercial accuracy, release approval, biometric identity functionality, or a product-performance claim. The remaining gates are: broader rights-cleared held-out evidence across face scale, pose, lighting, multiple faces and no-face negatives; packaging/native-dependency notices and redistribution review; long-run soak/resource/performance validation on representative supported hardware; and explicit commercial-release approval. Do not tune the frozen `0.50` threshold from the one admitted real source and do not add recognition, embeddings, ReID or identity matching.

### Donor-screen outcome
The first face donor screen remained fail-closed:
1. OpenCV Zoo YuNet was held out because its WIDER FACE training-data path did not meet the repository's commercial provenance standard.
2. Open Model Zoo `face-detection-retail-0004` was held out because model/training-data provenance was not disclosed strongly enough for the required code/weight/data separation.
3. MediaPipe BlazeFace was held out because the selected pretrained model's training-data/weight provenance did not close strongly enough for commercial shipping.

The official Google/TensorFlow FaceSSD MobileNetV2 Open Images V4 artifact was subsequently provenance-admitted and pinned, but its practical execution/conversion hypothesis is now **closed for this sprint**. Direct OpenCV import, OpenCV SSD rewrite, direct LiteRT, TensorFlow 1 legacy conversion, Coral preconverted lineage, LiteRT C++ SDK, stable `litert-converter`, `ai-edge-litert` Python, TensorFlow 2.21 pip/libclang, TensorFlow 2.21 TOCO source build, and OpenVINO full/cut GraphDef conversion have all been classified and must not be retried unchanged.

PR #104 merged the final negative OpenVINO evidence into `main` at `d6c790661c28126ec17e2745844c3704a63d8ac1`. Both full FaceSSD GraphDef conversion and the documented cut at `raw_outputs/box_encodings`, `raw_outputs/class_predictions`, and `anchors` failed in OpenVINO `2026.3.1` before CPU compilation. No inference, media, input tensor, or retained derived artifact occurred. This closes the FaceSSD/OpenVINO path and the broader FaceSSD practical execution hypothesis for the current sprint.

### Changed donor approach — standard OIDv4 SSD MobileNetV2
The active candidate is the official Google/TensorFlow Object Detection API standard Open Images V4 model `ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz`, not the quantized FaceSSD custom-postprocess graph. The model-zoo lineage is pinned to `tensorflow/models@0558408514dacf2fe2860cd72ac56cbdf62a24c0`, Apache-2.0, with LICENSE Git blob `43fcf7bf1f1f9f824a1debf05d6ced45bf5810aa`.

The corrected OIDv4 label map is pinned to `research/object_detection/data/oid_v4_label_map.pbtxt`, Git blob `643b9e8ed5d9239a3248b895fb32f3b51caa92f3`. `Human face` maps to class id `502`, MID `/m/0dzct`. The candidate archive contains no label-map member, so this exact corrected external label map is the authoritative mapping. Open Images V4 remains subject to Google's caveat: annotations are CC BY 4.0, images are listed CC BY 2.0, and each image license must be independently verified before any image is used for future training/evaluation.

Analytics quality run #259 was the bounded first discovery head and passed Linux, Windows and the aggregate gate. It downloaded the exact official Google archive only in memory, performed no extraction, inference, network construction, media processing or artifact retention, and measured:

- archive size `158,851,107` bytes;
- archive SHA-256 `8dd82cc52625eb9b43c6ed8050868fe1e78f878c6f028519fc731b3c7e9035f7`;
- seven regular files in exact tar order:
  1. `model.ckpt.meta` — 11,915,505 bytes — `541aebeef5c674609f4b2cc229c265b053dc7df3226143b01037c0a42d715f7f`;
  2. `checkpoint` — 77 bytes — `dd1b025d2e155283f5e300ce95bf6d5b6bc0f7fe010db73daa6975eb896ab9cb`;
  3. `frozen_inference_graph.pb` — 66,606,111 bytes — `150f3eb77b741ed24e1a19559589205ccd616b059fc976c8b8d5cefc126bf86b`;
  4. `saved_model/saved_model.pb` — 67,889,548 bytes — `f5453b9c2bb73be4d21eef9eb37a8fce0a2d09903ea1aecd44ff49fa6efe258f`;
  5. `model.ckpt.index` — 14,175 bytes — `0557da0b4b7d555fc1483d3ba3a89732c7606e7176c30839335333f48b9d81f4`;
  6. `pipeline.config` — 4,267 bytes — `cf424b06dabcc7acd6bf71ffd941c5a9890b975f8036444e03a2290391a24614`;
  7. `model.ckpt.data-00000-of-00001` — 57,841,536 bytes — `61bc5931d1cc44cd83b80ebee3cb75757e84c1a49ac5cea71149e36a93da4044`.

PR #105 then hard-pinned those archive/member identities, explicitly required the frozen graph, SavedModel graph and pipeline config, and failed closed if a label-map member unexpectedly appeared. Exact-head Analytics quality run #260 passed Linux, Windows, the bounded standard-donor admission evidence and the aggregate gate. PR #105 merged into `main` at `8267dd00bb645dbc41f76f2ee2e9542473b2bf14`; automatic main verification run #261 also passed. The standard OIDv4 artifact/provenance admission gate is therefore closed green. No model inference, media use, artifact retention or accuracy claim occurred in that admission.

### Standard OIDv4 runtime construction — closed green
PR #106 performed the smallest post-admission runtime-construction attempt with the exact admitted `frozen_inference_graph.pb` and OpenVINO `2026.3.1`. Exact-head Analytics quality run #262 passed Linux, Windows and the aggregate gate. OpenVINO's TensorFlow frontend loaded the graph and the CPU plugin compiled it successfully without inference, an input tensor, media use or retained derived artifact.

The verified model ports were:
- input `image_tensor:0` — uint8, `[?,?,?,3]`;
- boxes `Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack/TensorArrayGatherV3:0` — float32, `[?,100,4]`;
- classes `add:0` — float32, `[?,100]`;
- scores `Postprocessor/BatchMultiClassNonMaxSuppression/map/TensorArrayStack_1/TensorArrayGatherV3:0` — float32, `[?,100]`;
- count `Postprocessor/ToFloat_3:0` — float32, `[?]`.

PR #106 merged to `main` at `bb069a1899ce51a85b90d7a0ee6a7cc2cb1662a6`, and automatic main verification run #263 also passed. This is compatibility evidence only; it establishes neither face-detection accuracy nor commercial readiness.

### Standard OIDv4 synthetic execution — closed green
PR #107 added the class-502-only adapter and one no-media execution through the default-on privacy-blur path. Exact-head Analytics quality run #264 passed repository guardrails, Linux, Windows, the bounded synthetic face evidence and the aggregate gate on unchanged base `bb069a1899ce51a85b90d7a0ee6a7cc2cb1662a6`. PR #107 merged to `main` at `2829df3bb0dc5ac50eccc460715de8f95cb81b77`; automatic main verification run #265 also passed.

That smoke supplied one deterministic generated in-memory uint8 tensor to the admitted standard OIDv4 graph, admitted only class `502` (`Human face`) into the existing normalized `FaceDetection` contract, and exercised the default-on blur policy. No image/video source, real person, retained model/pixel artifact, tuning, recognition/ReID/identity matching, or face-detection accuracy/commercial-readiness claim occurred. The fixed pre-measurement face threshold remains `0.50`.

### Rights-cleared real-person source admission — closed green
PR #108 selected Wikimedia Commons `Face portrait (Unsplash).jpg`, source page `https://commons.wikimedia.org/wiki/File:Face_portrait_(Unsplash).jpg` and canonical upload `https://upload.wikimedia.org/wikipedia/commons/0/04/Face_portrait_%28Unsplash%29.jpg`, solely as the first real-person evidence source. Commons records the image as a William Stitt photograph dated 19 October 2016, published on Unsplash before the 5 June 2017 license change under CC0-1.0, with byte length `9,023,019` and SHA-1 `d7ac58077d135b823e71e7b38e763f082e2053bc`.

Discovery run #266 passed the repository guardrails, Linux regression, the bounded source-admission step, Windows regression and the aggregate quality gate. The source was streamed only in memory, never decoded, and produced SHA-256 `7356daa8fd4ad53b946ce0036f06b014431dc89b7ae29ecd8ef18fc54edce6b5`. The pinning head then hard-pinned and reverified that exact size/SHA-1/SHA-256; exact-head run #267 passed Linux, Windows, the bounded admission evidence and the aggregate gate. PR #108 merged into `main` at `13c3f9f7ee4e3b7a23e029557251e6bbce71a76e`, and automatic main verification run #268 also passed.

No image decode, face inference, face count, model execution, media retention, or detection-accuracy/commercial-readiness claim occurred during admission. The rights-cleared source is now admitted for the separate one-attempt face-detection -> automatic-blur measurement.

### First real-person face -> automatic-blur measurement — active one-attempt gate
The current acceptance-moving work item is one untouched execution of the admitted standard OIDv4 graph against the exact admitted CC0 portrait, through the class-502-only adapter and existing default-on privacy-blur path. The threshold is frozen at `0.50` before execution. The evidence lane must reverify both model and source identities, decode pixels only in memory, run exactly one CPU inference, preserve every returned class-502 score/normalized box, verify input-frame immutability, and record the privacy action and whether the output pixels changed.

There is no independently authored face ground-truth annotation for this source, so misses and false positives remain explicitly unscored rather than inferred from the source title or detector output. Zero detections, one detection, or multiple detections must all be preserved as obtained; none is grounds for threshold tuning or an unchanged rerun. No recognition, embeddings, ReID, identity matching, model/media retention, or commercial-accuracy claim is permitted. This gate is a single-source engineering observation only.

Runs #269 and #270 both passed the normal Linux regressions and reached the dedicated measurement step, then were cancelled by the repository's existing five-minute Linux job ceiling before any face result was emitted. Run #270 already changed the detector input to the model's pinned 300x300 inference size, so it was not eligible for an unchanged retry. Run #271 changed the acquisition path and made the previously opaque budget measurable: exact graph stream/verification completed in `1.549758s`, source stream/verification in `0.417205s`, image decode in `0.145742s`, and OpenVINO conversion+CPU compilation in `189.092199s`. The job then exhausted the remaining budget inside the combined inference+blur phase without emitting a result.

The next repaired head keeps the exact admitted model/source, fixed `0.50` threshold, all provenance/security checks and the five-minute ceiling. It preserves the run-#271 stream-to-exact-pinned-graph optimization, times the single detector inference separately from privacy processing, reuses that exact detection tuple so privacy enforcement cannot trigger a second inference, and bounds pathological full-resolution Gaussian kernel work by downscaling the detected face region before Gaussian blur and restoring it afterward while preserving the requested Gaussian strength in source-pixel units. This is a runtime-budget repair, not a detection result or accuracy claim. No model/media artifact is retained.

If the standard Google donor later fails on actual rights-cleared evidence, do not donor-shop indefinitely. Move to the project-owned training fallback only after the current donor path is conclusively closed: a permissive detector architecture such as the already-reviewed Apache-2.0 YOLOX-Nano style family plus individually rights-cleared Open Images Human-face training/validation data, with exact trainer/dependency/data/export lineage and no-spend/resource ceilings. Expensive training remains prohibited until the donor path is conclusively closed and the plan fits policy.

## Commercial-release gates
Nothing in the current face evidence supports a commercial accuracy claim. Artifact provenance, runtime compatibility, synthetic integration correctness, rights-cleared real-person accuracy evidence, privacy/security review, package/native dependency closure, notices, versioned integration and explicit owner commercial-release approval remain separate gates.

No home/customer media, credentials, biometric references, embeddings or retained face/model artifacts are permitted in CI. All evidence execution stays on GitHub-hosted runners inside the bounded branch families defined by `AGENTS.md`.

## Reproduce
```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```
