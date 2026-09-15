# OpenVINO Runtime engineering review

Reviewed 2026-09-15 for the optional local inference boundary only. This is engineering provenance, not a legal opinion or commercial release approval.

## Runtime identity

- Upstream: `https://github.com/openvinotoolkit/openvino`
- Reviewed release tag: `2026.3.1`
- Tag commit: `759c5a6ab8c066af5f4bc5ebd04643706012a37d`
- Upstream source license at that commit: Apache-2.0
- PyPI project: `openvino`
- PyPI release date for 2026.3.1: 2026-08-26
- PyPI license expression: Apache-2.0
- Python requirement reported by PyPI: Python >=3.10

The Analytics Lab adapter requires a runtime version string beginning with `2026.3.1`; this is an engineering compatibility/provenance gate, not proof of a particular wheel's integrity.

## API surface used

The adapter uses the documented Python API only:

1. `openvino.Core()`
2. `Core.read_model(model=<xml path>, weights=<bin path>)`
3. `Core.compile_model(model, <device>)`
4. `CompiledModel.inputs` / `CompiledModel.outputs` and output-port shapes
5. callable `CompiledModel` synchronous inference returning output mappings

The selected IR inputs and output shapes are checked against the already reviewed Open Model Zoo model documentation before output is accepted.

## Model post-processing boundary

The selected Open Model Zoo models are pinned separately at commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`.

- `person-detection-retail-0013` supplies normalized SSD rows for class `1` (person).
- `human-pose-estimation-0001` supplies 19 heatmap channels and 38 PAF channels.
- Analytics Lab initially uses detector-isolated person crops and only shoulder/hip heatmap peaks needed by its posture classifier.
- Keypoint channel IDs and the small quarter-pixel peak refinement are adapted from the Apache-2.0 Open Model Zoo `open_pose.py` decoder at the pinned OMZ commit.
- The full PAF grouping algorithm is not vendored in this baseline. This simplification is explicitly unvalidated and must be compared against labeled video before promotion.

## What is not approved yet

No OpenVINO wheel is vendored by this repository. Platform-specific wheel SHA-256 values, bundled/native third-party notices, installer behavior, update policy and final redistribution packaging remain release-review items. No model binary is committed to the public repository. A locally viewable or downloadable model is not automatically approved for product redistribution merely because the upstream project uses Apache-2.0; the existing artifact manifest and release review remain authoritative.

The runtime boundary does not authorize paid compute, GPU jobs, public webcam retention, customer/home footage, production deployment, or commercial-release claims.
