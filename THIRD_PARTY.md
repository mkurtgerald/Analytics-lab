# Third-Party Component Register

Every external library, repository, model, weight file, dataset, code fragment, or other donor component used by Analytics Lab must be recorded here before it is merged into a production candidate.

## Plate-specialist OCR donor candidate

The optional CCT recognizer adapts preprocessing and fixed-slot decoding from
[ankandrew/fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) at
`9ce7a5b64a939aa421c243b331d42e6bc25ffd44`, especially
`fast_plate_ocr/core/process.py` and `inference/plate_recognizer.py`.
The publisher distributes the project and associated model release under its
[MIT notice](https://github.com/ankandrew/fast-plate-ocr/blob/9ce7a5b64a939aa421c243b331d42e6bc25ffd44/LICENSE),
retained in [notices/fast-plate-ocr-MIT.txt](notices/fast-plate-ocr-MIT.txt).
No upstream package, training code or model bytes are vendored.

The selected publisher `arg-plates` release artifact is
`cct_s_v2_global.onnx`, 5,262,230 bytes, SHA-256
`384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b`.
Its `cct_s_v2_global_plate_config.yaml` is 1,725 bytes, SHA-256
`0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6`.
Both sizes/digests were independently read from the publisher's release API;
config contents were checked against that digest before implementing the
RGB uint8 64x128 input and ten-slot Latin alphabet contract. OpenVINO's existing
ONNX frontend loads this exact model; ONNX Runtime and training frameworks are
not added. The existing runtime versions use official hash-locked Windows wheels.

Code, model distribution and evaluation-media provenance remain distinct. The
publisher's global training corpus is private and its overlap with our two
exposed CC0 plate examples is unknown. This admission supports bounded local
engineering evaluation only, not training-corpus acquisition, weight/model
redistribution, production packaging, default promotion or a field accuracy
claim. Keep all donor and transitive notices for later release review. Published
donor validation rates and GPU timings are not Analytics Lab measurements.

The paired evidence path reuses the separately admitted CC0 Land Rover and South
Carolina sources, pre-OCR crop hashes and exact expected texts, plus three
explicitly generated blank controls. It does not acquire the held Washington
source or a new dataset. The official Tesseract installer and Apache-2.0 English
model retain their existing separate identities and evidence-only restrictions.

## Bounded Caltrans public live evaluation source

Caltrans public CWWP2 district 1 camera index `88`, `US-101 : Eureka / 14th -
Looking North (C039)`, exposes a 320x240 HLS feed at
`https://wzmedia.dot.ca.gov/D1/eureka_14th_320x240.stream/playlist.m3u8` through
its [public catalog](https://cwwp2.dot.ca.gov/data/d1/cctv/cctvStatusD01.json).
Primary review on 2026-10-18 binds the
[CWWP2 purpose](https://dot.ca.gov/programs/traffic-operations/traveler-information/cwwp),
[CCTV access and fair-use guidance](https://cwwp2.dot.ca.gov/closed-circuit-television-cameras.html)
and [ownership terms](https://dot.ca.gov/conditions-of-use).
The public portal offers CCTV for public/commercial integration and treats
Caltrans-created information as public domain unless otherwise indicated;
third-party and contrary notices remain exceptions. Its public website delivery
is distinct from the direct TMC subscription described in TOPP97-1. The selected
catalog entry has no contrary/third-party notice. Keep attribution to Caltrans,
avoid service-degrading access and perform no face/plate identification.

Admission is limited to one short non-identifying traffic compatibility test in
the named hosted evidence lane: the latest two consecutive unencrypted segments
from a verified three-entry playlist, at most
35 seconds declared media and 4 MiB, hashed before local decoding and deleted
with models/frames on success or failure. Runtime byte hashes identify the
actual dynamic material; a static camera URL or catalog modification date is not
an immutable frame identity or capture timestamp. No training, archival dataset,
redistribution, standing capture service, release or accuracy admission follows.
The existing OMZ 0106 vehicle head (class 1) is reused unchanged at 0.50 with the
admitted runtime; its published front-facing barrier/BIT-Vehicle results do not
qualify this wide traffic scene. ByteTrack handles are segment-local only.

## Policy
Preferred licenses: Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause. Copyleft, non-commercial, research-only, custom or unclear terms require explicit review. Code, weights and data rights are separate admissions.

## Admitted engineering components
| Component | Upstream URL | Version / Commit | License | Purpose | Modified? | Redistribution Requirements | Approval |
|---|---|---|---|---|---|---|---|
| Open Model Zoo `person-detection-retail-0013` FP16 artifact manifest | `https://github.com/openvinotoolkit/open_model_zoo` | `6697dead54ed1cdd664b0313189c2cb52ee6335e` | Apache-2.0 | Candidate local person-detector artifact identity | No donor code/model vendored | Preserve applicable Apache-2.0 license/notices | Engineering local-evaluation baseline; release review pending |
| Open Model Zoo `human-pose-estimation-0001` FP16 artifact manifest | `https://github.com/openvinotoolkit/open_model_zoo` | `6697dead54ed1cdd664b0313189c2cb52ee6335e` | Apache-2.0 | Candidate local pose artifact identity | No donor code/model vendored | Preserve applicable Apache-2.0 license/notices | Engineering local-evaluation baseline; release review pending |
| Open Model Zoo `person-detection-0200` FP16 | `https://github.com/openvinotoolkit/open_model_zoo/tree/6697dead54ed1cdd664b0313189c2cb52ee6335e/models/intel/person-detection-0200` | Same pinned OMZ commit; exact XML/BIN sizes and SHA-384 in `OPENVINO_OMZ_PERSON_0200_FP16` | Apache-2.0 | Existing detector-shootout donor now selectable through the reusable local backend; input 1x3x256x256 BGR, person class 0 | No weights/code vendored; project-owned adapter | Preserve Intel/Open Model Zoo license/notices; training/dependency/redistribution release review remains separate | Bounded engineering comparison only, default unchanged; no commercial accuracy or release admission |
| Open Model Zoo OpenPose decoder reference | `https://github.com/openvinotoolkit/open_model_zoo` | `6697dead54ed1cdd664b0313189c2cb52ee6335e` | Apache-2.0 | Reviewed keypoint channels / evidence-only diagnostic | Adapted diagnostic code | Preserve Intel Apache-2.0 attribution/notice | Engineering evidence diagnostic only |
| OpenVINO Runtime Python API | `https://github.com/openvinotoolkit/openvino` | tag `2026.3.1`, commit `759c5a6ab8c066af5f4bc5ebd04643706012a37d` | Apache-2.0 | Local CPU inference/runtime-construction family used by bounded evidence | Not vendored | Platform wheel/native dependency notices must be pinned before product packaging | Engineering API baseline; FaceSSD conversion path closed, standard OIDv4 SSD construction may be separately bounded after artifact admission |
| FoundationVision/ByteTrack portable association slice | `https://github.com/FoundationVision/ByteTrack` | `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` | MIT | Detector-neutral two-stage association | Yes, adapted slice with project-owned stdlib maximum-cardinality/maximum-weight assignment; no native LAP/Kalman dependency added | Preserve Yifu Zhang 2021 MIT notice | Engineering integration baseline; no full-upstream parity claim; real-video acceptance pending |
| Open Model Zoo `vehicle-license-plate-detection-barrier-0106` FP16 | `https://github.com/openvinotoolkit/open_model_zoo` | `6697dead54ed1cdd664b0313189c2cb52ee6335e`; XML 325,452 bytes SHA-384 `b0a70cfc4ccbe85dd5ba5549c2d142d7f922a0a1116657cdcc9f48cb6afc7cc40a596a655b9e6566396c48b72232ff65`; BIN 1,286,772 bytes SHA-384 `0406e296c9822f4a5e8ee300ad40e5c0a0eeeaf9206f829556cb836cb8f10a4f8a4dd94d1bb16a60562bf1568fd12618` | Apache-2.0 | First replaceable plate-detection baseline; now under replacement after documented-envelope miss | Runtime adapter only | Preserve Intel/Open Model Zoo notices | Engineering-only admission; not retained as accepted commercial detector |
| Tesseract OCR | `https://github.com/tesseract-ocr/tesseract` | `5.5.3`, commit `db0ec62f81b0737fbbe184d8fea40af5738f8eef` | Apache-2.0 | Replaceable Latin OCR engine | CLI adapter only | Preserve Apache-2.0 notices; Leptonica/image-codec native packaging review pending | Engineering runtime baseline |
| Tesseract 5.5.3 official Windows release installer | `https://github.com/tesseract-ocr/tesseract/releases/tag/5.5.3` | `tesseract-ocr-w64-setup-5.5.3.20260724.exe`; 26,573,224 bytes; SHA-256 `bee9e3434bd94fd65387d9be28cd467a41f61b1275383b55b0f59a1331270ae4` | Tesseract source is Apache-2.0; bundled native dependency licenses remain a separate packaging review | Exact hosted execution artifact for one bounded OCR evidence attempt | Not vendored; downloaded ephemerally from the upstream release | Do not redistribute as a product artifact until bundled dependency notices/rights are closed | Engineering evidence execution only; product packaging not approved |
| Tesseract `tessdata_fast` English model | `https://github.com/tesseract-ocr/tessdata_fast` | `87416418657359cb625c412a48b6e1d6d41c29bd`; `eng.traineddata` 4,113,088 bytes; Git blob `bbef4675053b5b468cdb477053e28b1c698ba08e` | Apache-2.0 | English/Latin OCR model | Not vendored | Preserve Apache-2.0 notices; exact corpus/font provenance remains release gate | Engineering baseline only |
| OpenCV morphology/runtime for model-free plate proposals | `https://github.com/opencv/opencv` and `https://github.com/opencv/opencv-python` | OpenCV tag `4.12.0` -> commit `49486f61fb25722cbcf586b7f4320921d46fb38e`; opencv-python tag `88` -> commit `fa742a47d3993e45502dff54d96af6a4efb65153`; evidence wheel `opencv-python-headless==4.12.0.88` | OpenCV Apache-2.0; packaging repository MIT | Model-free horizontal-gradient/morphology proposal baseline after measured OMZ miss | Project-authored proposal logic using public OpenCV APIs; no donor algorithm copied | Preserve applicable OpenCV/package notices; exact wheel/native codec notices and hashes must be closed before product redistribution | Engineering evidence baseline only; no weights/training data; commercial release review pending |

## Reviewed alternatives / not incorporated
| Candidate | Exact revision | Repository license | Fit | Status / blocker |
|---|---|---|---|---|
| `ankandrew/open-image-models` plate detectors | `f22000e02b30642f317cdba7755c0631638b109e` | MIT repository code | Pretrained ONNX plate detection | Rejected from shippable path for now: release model asset license, exact training-data provenance/rights and immutable artifact identity are not sufficiently established. Repository license is not treated as weight/data permission. |
| `LdDl/license_plate_recognition` | `ccc2a4ca82845d13070326b2b7104785c484603f` | Apache-2.0 repository code | Published ONNX plate detection/OCR weights | Rejected from shippable path for now: README states training on a Russian plate dataset, but exact dataset commercial rights/provenance and released-weight redistribution terms are not pinned separately. |
| `roboflow/rf-detr` architecture fallback | `d96d6ff9303ec5af7bb8f73e83f0f04ecb6ef728` | Apache-2.0 repository code; LICENSE blob `56db37c53b61c7976042682a83d85383a6bf6007` | Preferred project-owned Weapons training architecture candidate after measured OIDv4 rifle miss | Architecture source only. Stock COCO weights do not supply the required firearm taxonomy and are not admitted as the Weapons model. Any future training must use `RFDETRNano` with `pretrain_weights=None` for the first offline proof and separately rights-cleared media/annotations under the repository training-manifest gate. |
| Google/TensorFlow OIDv4 SSD ResNet101-FPN, 512x512 | Model-zoo entry at `tensorflow/models@0558408514dacf2fe2860cd72ac56cbdf62a24c0`; official archive `ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20.tar.gz` | TensorFlow Models source Apache-2.0; selected weights/data rights and immutable artifact identity remain separate | Higher-resolution pretrained Weapons comparison candidate using the existing runtime family | Not incorporated or admitted. Official Google storage HEAD returned Content-Length 638,212,759 on October 11, 2026. No archive download, hash discovery, construction or inference occurred. This exceeds the existing bounded model-discovery lane; exact weight/runtime/resource admission and independent evaluation are required. Upstream whole-dataset metrics do not establish Weapons improvement. |
| `PaddlePaddle/PaddleOCR` | `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf` | Apache-2.0 repository code | General text detection/OCR candidate | Not admitted: would add a second large runtime/model family without exact selected-model artifact, training-data rights, redistribution closure or measured need beyond the current smaller model-free alternative. |
| tryolabs/norfair | `e517b4236f6b67a6ecf342f5df1fccb7788dbc54` | BSD-3-Clause | Detector-agnostic tracking | Fallback only; dependency/runtime review incomplete. |
| Open Model Zoo multi-camera multi-target tracking demo | `6697dead54ed1cdd664b0313189c2cb52ee6335e` | Apache-2.0 | Person/vehicle ReID tracking | Heavier and ReID-coupled; future appearance-search reference only. |
| FoundationVision/ByteTrack full Kalman/LAP path | `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` | MIT donor code; transitive review required | Motion/global assignment | Do not admit unless real tracking evidence demonstrates a specific need. |
| OpenCV Zoo YuNet face detector | `opencv/opencv_zoo@47534e27c9851bb1128ccc0102f1145e27f23f98`; 2023mar ONNX LFS SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`, 232,589 bytes | MIT model-directory license | Lightweight FaceDetectorYN-compatible detector | Not admitted to shippable path: upstream training repo `ShiqiYu/libfacedetection.train@a61a428929148171b488f024b5d6774f93cdbc13` explicitly trains on WIDER FACE; commercial dataset/training-rights closure is insufficient under fail-closed policy. |
| Open Model Zoo `face-detection-retail-0004` FP16 | `6697dead54ed1cdd664b0313189c2cb52ee6335e`; XML SHA-384 `a7f8d1d41998503c4f3cdd8c12275f04f1736e5142127edcb4c76c3e17188499390574095a5b2a9dd78d3d0f77d02034`; BIN SHA-384 `394185d3e42c34d7f9d43229ec8f5755c07e19fd6469d23883e71707fdd8eb66d90ff3ba1c94adac599b` | Apache-2.0 OMZ license | Compact front-facing face detector | Not admitted: artifact/license identity is clear, but training-data provenance is not disclosed strongly enough for required commercial code/weight/data separation. |
| MediaPipe BlazeFace short-range | `google-ai-edge/mediapipe@8ac5a39c659578c2595a54ef5277608173c217fe` | Apache-2.0 repository | Cross-platform face detector candidate | Not admitted: selected pretrained model training-data and weight-provenance closure is insufficient for the commercial shipping gate. |
| Google/TensorFlow FaceSSD MobileNetV2 quantized Open Images V4 | TensorFlow model zoo at `tensorflow/models@8b12ae202a3ccf8f965c730a4e7617204e32000b`; archive `facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz`; archive 130,655,026 bytes SHA-256 `9ae49a245caddbe7d7bbc82a35da0191a2f2e210161df19be357a1c7f49118d5`; members: `face_label_map.pbtxt` 56 bytes SHA-256 `87f1e97ff18442302ca7686276e59beac925126b91b14cb15dee710de2ee9c60`, `model.ckpt.data-00000-of-00001` 86,462,816 bytes `693c0eb84b8d9349391d66c38d2ac03b3d628029eb9f9ad7417302f6ac355599`, `model.ckpt.index` 68,838 bytes `e00ccd57873134cc9a8a24cf2f39a16ee59a6037012755564639e06197c15729`, `model.ckpt.meta` 21,373,388 bytes `65346dc8d8f11297df102c6e134120d3a248dd3b7e38ec0fcf3799f49d137c28`, `pipeline.config` 4,829 bytes `f51b55181cf8a614c75ffe716b5c5a6f253ca5037199f37440775a925482f224`, `tflite_graph.pb` 22,222,216 bytes `dc8e2c9e21407b2f6d35f1eb655ba8a0c9c73094e5987231a1a8de2edae74978`, `tflite_graph.pbtxt` 62,525,550 bytes `e1232ff66eedd5676bfa31e78aba28ee6244b0322a5646d1a1f89efa02cb781b` | Apache-2.0 TensorFlow models distribution; Open Images annotations CC BY 4.0 and source images listed CC BY 2.0 | Provenance-admitted face donor | Immutable artifact identity remains closed, but the practical execution/conversion hypothesis is closed for this sprint after direct OpenCV, rewrite, LiteRT, TF1/TF2 converter, Coral lineage, and OpenVINO full/cut routes failed or failed bounded admission. Do not retry unchanged. |
| Google/TensorFlow standard OIDv4 SSD MobileNetV2 | `tensorflow/models@0558408514dacf2fe2860cd72ac56cbdf62a24c0`; archive `ssd_mobilenet_v2_oid_v4_2018_12_12.tar.gz`, 158,851,107 bytes, SHA-256 `8dd82cc52625eb9b43c6ed8050868fe1e78f878c6f028519fc731b3c7e9035f7`; regular members in tar order: `model.ckpt.meta` 11,915,505 bytes `541aebeef5c674609f4b2cc229c265b053dc7df3226143b01037c0a42d715f7f`, `checkpoint` 77 bytes `dd1b025d2e155283f5e300ce95bf6d5b6bc0f7fe010db73daa6975eb896ab9cb`, `frozen_inference_graph.pb` 66,606,111 bytes `150f3eb77b741ed24e1a19559589205ccd616b059fc976c8b8d5cefc126bf86b`, `saved_model/saved_model.pb` 67,889,548 bytes `f5453b9c2bb73be4d21eef9eb37a8fce0a2d09903ea1aecd44ff49fa6efe258f`, `model.ckpt.index` 14,175 bytes `0557da0b4b7d555fc1483d3ba3a89732c7606e7176c30839335333f48b9d81f4`, `pipeline.config` 4,267 bytes `cf424b06dabcc7acd6bf71ffd941c5a9890b975f8036444e03a2290391a24614`, `model.ckpt.data-00000-of-00001` 57,841,536 bytes `61bc5931d1cc44cd83b80ebee3cb75757e84c1a49ac5cea71149e36a93da4044`; corrected label map `research/object_detection/data/oid_v4_label_map.pbtxt` Git blob `643b9e8ed5d9239a3248b895fb32f3b51caa92f3`; Human face id 502 / MID `/m/0dzct`; LICENSE Git blob `43fcf7bf1f1f9f824a1debf05d6ced45bf5810aa` | Apache-2.0 TensorFlow Models distribution; Open Images annotations CC BY 4.0 and source images listed CC BY 2.0 with per-image verification caveat | Changed face-detection donor candidate with standard frozen graph | Artifact/provenance identities are pinned and must reverify on exact-head hosted CI before merge. Archive has no label-map member, so the exact corrected external label map is authoritative. Runtime compatibility, product packaging/native notices, any real-person evidence and all accuracy claims remain separate gates. No model is vendored or retained. Weapon-lane reuse is limited to the same admitted artifact/runtime and the pinned OIDv4 label-map classes Knife 285, Kitchen knife 325, Rifle 351, Shotgun 361, Sword 365, Weapon 408, and Handgun 533; this class allowlist is an integration boundary only and carries no accuracy claim. |
| Google LiteRT Python runtime, direct FaceSSD GraphDef execution candidate | `google-ai-edge/LiteRT@145c7523ff08d5e57ab5c582141775eea47da9c7` (`v2.2.0`); PyPI `ai-edge-litert==2.2.0`; CPython 3.11 Windows wheel SHA-256 `c36ccb82a89f10f9e6c947f67f06dd00699caf5fe3bcb2be0eddcb213f70220d`; Linux x86-64 wheel SHA-256 `0171f556fffa335281acd7d468de999fd4de32f6bf1a5f14e680676591cef764` | Apache-2.0 source; selected transitive/native package notices would require separate closure | Official CPU runtime candidate for Windows/Linux | FaceSSD direct path closed; do not retry unchanged. |
| TensorFlow 1.15.5 source TFLite converter path | `tensorflow/tensorflow@3db52be7be81a87c623cdeb7f03d3767521c5246` (`v1.15.5`); matching FaceSSD instructions at `tensorflow/models@8b12ae202a3ccf8f965c730a4e7617204e32000b` | Apache-2.0 source | Historical official FaceSSD GraphDef conversion path | Closed before execution because the legacy Bazel/Python/dependency stack does not clear the bounded exact-toolchain gate. Do not retry unchanged. |
| Google Coral preconverted MobileNet SSD v2 face FlatBuffer | `coral/edgetpu@99c63d73b84fb54bab7b1ddfde7a8f1a3200c5f0`; `test_data/mobilenet_ssd_v2_face_quant_postprocess.tflite`, 5,504,160 bytes, Git blob SHA-1 `de3075475a0022f1b6fda055731620535c7ff6a3` | Apache-2.0 repository; model lineage/weight provenance separately gated | Potential preconverted FaceSSD route | Closed for this sprint because exact authoritative lineage to the admitted FaceSSD donor was not established. Do not donor-shop or retry unchanged. |
| OpenCV TensorFlow SSD text-graph helpers | `opencv/opencv@49486f61fb25722cbcf586b7f4320921d46fb38e`; `tf_text_graph_ssd.py` 18,314 bytes Git blob `d27fd0d384f509789bf64f069203d3f9964db576`; `tf_text_graph_common.py` 10,055 bytes Git blob `c82053b4fb05aa2fd40c45b1cefa550431608847` | Apache-2.0 | Evidence-only FaceSSD graph rewrite/compatibility classification | Route closed after bounded classification; not production packaging approval. |
| TensorFlow 2.21.0 native TOCO converter target | `tensorflow/tensorflow@a481b10260dfdf833a1b16007eead49c1d7febf3`; Bazel `7.7.0`; public target `//tensorflow/lite/toco:toco` | Apache-2.0 source | FaceSSD frozen GraphDef -> TFLite conversion candidate | Closed before build: target pulls TensorFlow core/lib/ops/protos, TFLite import/export, protobuf, Abseil and TOCO transformation stack; not a bounded standalone converter under five-minute hosted CI/toolchain policy. Do not retry unchanged. |

## GMDCSA-24 retrospective activity-label provenance

The already reviewed MIT evaluation source is
`ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos` at
`5abac7693229900cf80f722e878fbb119211fc1c`, by Ekram Alam, Abu Sufian,
Paramartha Dutta, Marco Leo and I. A. Hameed (Data in Brief,
DOI `10.1016/j.dib.2024.110892`). The existing source-rights registration and
upstream MIT notice remain applicable; no media or complete dataset is vendored.

The retrospective Subject-4 replay additionally pins publisher annotation
metadata: `Subject 4/Fall.csv`, 2,399 bytes, Git blob
`31c1625bbf51eff2e5bd767455c400f50f5786e4`, and `Subject 4/ADL.csv`,
3,092 bytes, Git blob `4d97c1ba3b3a43290529ae5c7be236322706816d`.
Their SHA-256 identities and the exact selected rows are recorded in
`analytics_lab.person_down_regression_evidence`. Falling/exercise activity
labels are source-authored evaluation proxies, not clinical person-down truth,
an injury diagnosis or a commercial acceptance claim. Subject 4 was previously
used, so this replay provides no new held-out generalization evidence.

## Rules
1. Never copy donor code without recording its source and license.
2. Preserve required copyright and attribution notices.
3. Treat model weights and datasets separately from source-code licenses.
4. Public source is not equivalent to commercial reuse permission.
5. Do not commit third-party datasets or weights unless redistribution rights are clear.
6. Prefer replaceable adapters around donor components.
