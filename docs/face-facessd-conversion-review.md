# FaceSSD conversion-path review

Date: 2026-09-22

## Decision

Close the historical TensorFlow 1.15 source-conversion hypothesis before execution. The already-admitted Google FaceSSD donor remains valid, but Google's matching TensorFlow Models instructions require a separate legacy TensorFlow Lite conversion step before LiteRT can consume the frozen `tflite_graph.pb`. That conversion route does not satisfy this repository's current smallest-runtime, exact-dependency and bounded-hosted-runner gate without introducing a materially larger legacy build toolchain.

This is a converter/toolchain classification, not a face-detector rejection. No TensorFlow runtime or converter was installed, no model member was materialized, no conversion or inference ran, no media was accessed, and no model/runtime artifact was retained or uploaded.

## Exact upstream producer contract

The admitted donor remains bound to `tensorflow/models@8b12ae202a3ccf8f965c730a4e7617204e32000b`. Its matching `research/object_detection/g3doc/running_on_mobile_tensorflowlite.md` is explicitly a TensorFlow 1.15 workflow. It says the SSD TensorFlow Lite path requires building TensorFlow from source and installing Bazel. The documented sequence exports `tflite_graph.pb`, then invokes the TensorFlow Lite converter through Bazel with the v1 converter to produce a FlatBuffer such as `detect.tflite`.

The documented quantized conversion interface binds:

- input graph: `tflite_graph.pb`;
- input tensor: `normalized_input_image_tensor`;
- input shape: `1,300,300,3`;
- output tensors: the four `TFLite_Detection_PostProcess` outputs;
- quantized uint8 inference;
- mean/std-dev 128/128;
- custom ops enabled.

This confirms PR #89's boundary: the admitted `.pb` is a frozen GraphDef, not the interpreter-ready FlatBuffer.

## Exact legacy TensorFlow toolchain reviewed

The latest TensorFlow 1.15 patch release is `tensorflow/tensorflow@3db52be7be81a87c623cdeb7f03d3767521c5246` (`v1.15.5`), Apache-2.0. Its exact `tensorflow/tools/pip_package/setup.py` exposes the `tflite_convert` console entry point but declares a broad legacy dependency set, including version-ranged `absl-py`, `astor`, `google_pasta`, `keras_applications`, `keras_preprocessing`, `numpy >=1.16.0,<1.19.0`, `opt_einsum`, `six`, `protobuf`, `tensorboard >=1.15.0,<1.16.0`, `termcolor`, `wheel`, `wrapt`, `h5py <=2.10.0`, and `grpcio`, plus exact `gast==0.2.2` and `tensorflow-estimator==1.15.1`.

At the same exact revision, `configure.py` admits Bazel only from `0.24.1` through `0.26.1`. The published TensorFlow 1.15.5 Python classifiers stop at Python 3.7, while this repository's hosted Linux and Windows regression jobs are fixed to Python 3.11 with five-minute job ceilings.

The matching FaceSSD documentation is stronger than the mere existence of an old TensorFlow wheel: for this SSD conversion path it explicitly requires a TensorFlow source build. Therefore installing an old wheel, bypassing its supported-Python envelope, using `--no-deps`, or silently substituting a newer converter would not reproduce the reviewed producer contract.

## Result

`legacy_tf115_conversion_path_admitted = false`

Reason: the exact official path expands the face sprint into a legacy source-build/Bazel stack with broad unresolved package selections and a Python/toolchain envelope outside the current hosted CI lane. That does not clear the repository's bounded dependency/provenance/package gate. Running it simply to observe a predictable environment/toolchain failure would add cost without acceptance value.

Do not retry the TensorFlow 1.15/Bazel conversion route unchanged. Do not weaken dependency pinning, move to an unreviewed converter, or treat a format conversion as equivalent merely because it can emit a `.tflite` filename.

## Smaller official preconverted candidate discovered

A safer change of approach exists and remains read-only at this stage. Google's Coral `edgetpu` repository at exact commit `99c63d73b84fb54bab7b1ddfde7a8f1a3200c5f0` contains `test_data/mobilenet_ssd_v2_face_quant_postprocess.tflite` as a 5,504,160-byte binary with Git blob SHA-1 `de3075475a0022f1b6fda055731620535c7ff6a3`. The commit was authored by Google and the repository carries Apache-2.0. The same test-data tree separately documents its Open Images V4 test-image licensing.

This file is only a candidate, not admitted evidence. Its filename and official Google origin make it a materially smaller next hypothesis than reconstructing the legacy converter stack, but the repository has not yet proved that this exact FlatBuffer is the preconverted derivative of the already-admitted FaceSSD donor. It also has not yet bound the binary's SHA-256 or its internal tensor/model identity in this project.

## Next smallest gate

Fail closed on lineage before any execution. Establish from authoritative Google evidence that the exact Coral FlatBuffer corresponds to the same FaceSSD MobileNetV2/Open Images V4 model family already admitted here. If lineage clears, add one admission-only hosted lane that downloads exactly this 5,504,160-byte file from the immutable Google source, verifies the Git-backed source identity and a newly discovered SHA-256, inspects only FlatBuffer metadata/tensor signatures, retains nothing, and performs no inference or media processing. Only after that artifact gate closes should the already-reviewed LiteRT CPU runtime be reconsidered for no-media construction/invocation smoke.

If exact lineage cannot be proven, stop this candidate rather than inferring equivalence from its filename or weakening the provenance rule.

## Safety boundaries preserved

No home/customer media. No public-person media. No face recognition, embeddings, ReID or identity matching. No paid resource. No model/runtime artifact retention. No accuracy claim. The merged default-on blur and permission-gated unblur path remains the product privacy boundary while executable face detection is unresolved.
