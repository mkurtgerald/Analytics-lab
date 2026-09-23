# FaceSSD direct LiteRT runtime review

Date: 2026-09-22

## Decision

Stop the direct LiteRT runtime hypothesis before installation or execution. The official Google LiteRT runtime is a legitimate CPU-capable Windows/Linux candidate, but the already-admitted FaceSSD artifact is not a LiteRT/TFLite FlatBuffer. The donor member `tflite_graph.pb` is the frozen TensorFlow graph that Google's matching Object Detection documentation says must be converted into a `.tflite` FlatBuffer before a TensorFlow Lite/LiteRT interpreter can run it.

This is a closed interface boundary, not a detector failure. No runtime was installed, no model member was materialized, no inference occurred, no media was accessed, and no model/runtime artifact was retained or uploaded.

## Exact runtime candidate reviewed

- Upstream: `google-ai-edge/LiteRT`
- Stable release: `v2.2.0`
- Exact release commit: `145c7523ff08d5e57ab5c582141775eea47da9c7`
- Source license: Apache-2.0
- Python distribution: `ai-edge-litert==2.2.0`
- Required project lane: CPython 3.11, CPU only; no GPU/NPU extras.
- Windows x86-64 wheel: `ai_edge_litert-2.2.0-cp311-cp311-win_amd64.whl`, 17.9 MB, SHA-256 `c36ccb82a89f10f9e6c947f67f06dd00699caf5fe3bcb2be0eddcb213f70220d`.
- Linux x86-64 wheel: `ai_edge_litert-2.2.0-cp311-cp311-manylinux_2_27_x86_64.whl`, 21.3 MB, SHA-256 `0171f556fffa335281acd7d468de999fd4de32f6bf1a5f14e680676591cef764`.
- Official platform matrix lists CPU support for both Linux and Windows.

The exact `v2.2.0` wheel-build source declares these direct Python requirements: `backports.strenum`, `flatbuffers`, `numpy >= 1.23.2`, `tqdm`, `typing-extensions`, `protobuf`, and `ml_dtypes`. Optional NPU and model-utils extras are unnecessary for the CPU interpreter path and were not admitted.

The direct requirements are not fully immutable as published because upstream leaves them version-ranged/unpinned. One compatibility detail is already material for this repository's Python 3.11 lane: `backports.strenum` 1.3.1 intentionally excludes Python 3.11+, while the older 1.2.8 wheel permits Python 3.11 and has SHA-256 `fc297cb26971f7d5e15a478a06a78575197f81daea47975771b1aae996dcccf4` under MIT. Because the donor/runtime interface is already incompatible, this review deliberately stops before selecting and hashing the remaining transitive wheels or attempting to infer redistribution closure from broad upstream dependency ranges.

The Python wheels bundle native LiteRT binaries (`.so`/`.dll`/`.pyd` through the official wheel packaging). Apache-2.0 is established for LiteRT source, but exact product redistribution notices for the complete selected native wheel plus every selected transitive package remain a separate packaging gate. No claim is made that the wheel bundle is product-redistribution-cleared merely because the upstream source is Apache-2.0.

## Exact donor interface evidence

The admitted donor remains unchanged:

- archive `facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz`
- 130,655,026 bytes
- SHA-256 `9ae49a245caddbe7d7bbc82a35da0191a2f2e210161df19be357a1c7f49118d5`
- exact frozen graph member `tflite_graph.pb`
- 22,222,216 bytes
- SHA-256 `dc8e2c9e21407b2f6d35f1eb655ba8a0c9c73094e5987231a1a8de2edae74978`

At the exact admitted TensorFlow Models revision `8b12ae202a3ccf8f965c730a4e7617204e32000b`, Google's `running_on_mobile_tensorflowlite.md` states that `export_tflite_ssd_graph.py` produces `tflite_graph.pb` / `tflite_graph.pbtxt`, then requires a separate TFLite Converter step to convert that frozen graph to a FlatBuffer such as `detect.tflite`. The same document describes the resulting `.tflite` file as the artifact run by the interpreter.

Current LiteRT documentation likewise constructs the Python interpreter from a `.tflite` model path or equivalent FlatBuffer bytes. Therefore renaming the admitted `.pb`, bypassing format validation, or installing LiteRT and feeding the GraphDef to it would not be a legitimate runtime test.

## Result

`direct_litert_runtime_compatible = false`

Reason: required model-format conversion has not occurred. This result is established from exact upstream producer/consumer contracts, so a no-media runtime failure is not needed merely to restate the known format mismatch.

Do not retry the direct LiteRT interpreter against `tflite_graph.pb` unchanged. Do not use `--no-deps`, ignore package metadata, or silently add a converter stack.

## Next smallest gate

Change approach to the conversion boundary. Review the exact official conversion path needed to transform the already-admitted frozen `tflite_graph.pb` into a TFLite FlatBuffer while preserving immutable donor identity and commercial provenance. The historical matching Google instructions use the TensorFlow Lite converter and explicitly identify TensorFlow 1.15-era tooling/source-build requirements, so that converter/runtime stack must receive its own dependency, license, platform, notice, and reproducibility review before any conversion is executed.

If that conversion stack cannot clear the same fail-closed package/provenance bar within the bounded hosted-runner budget, stop it and seek an official, immutably identifiable preconverted derivative of the same admitted donor rather than weakening the bar or switching to an unreviewed face model.

## Safety / evidence limits preserved

No home/customer media. No public-person media. No recognition, embeddings, ReID, or identity matching. No model/runtime artifact retention. No accuracy claim. The merged default-on blur and permission-gated unblur behavior remains the privacy boundary while detector execution is unresolved.
