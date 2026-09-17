# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path remains:

`authorized local video -> reviewed detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed model-artifact identities, runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video.

## Current acceptance-moving work
Live `main` entering this work item is `ebc3953b492d78a3ce0dc92f2cc2c1db7d5f5cc1`; its Analytics quality run `35279180587` passed on attempt 1. At intake there were zero open implementation PRs and zero active runs for that head.

The single work item is an evidence-only comparison of the current simplified `human-pose-estimation-0001` interpretation against the pinned Open Model Zoo reference semantics on the exact same rights-cleared two-clip seed. The diagnostic keeps the measured `person-detection-0200` low-threshold continuity selector, runs pose on the full frame using preserve-aspect resize plus right padding, applies 3x3 keypoint NMS and PAF grouping adapted from the pinned Apache-2.0 Open Model Zoo decoder, and then associates one decoded pose to the continuity-selected person box. The production candidate backend is unchanged.

### Measured detector baseline
The reviewed `person-detection-0200` detector at confidence 0.10 plus evidence-only spatial continuity recovered the low-confidence prone-person trajectory on this deliberately tiny staged-real seed:

- Positive fall interval: **96/96 frames = 100% coverage**; **94/95 transitions = 98.95% linked**.
- Positive overall: **173/174 frames = 99.43% coverage**; **1 reset / 171 transitions = 0.58%**.
- Hard negative: **212/212 frames = 100% coverage**; **211/211 transitions = 100% linked**; **0 resets**.

Threshold lowering alone was rejected because duplicate/noisy boxes rose sharply. The continuity selector is not a general detector promotion; the current seed is single-person and does not prove crowded-scene identity separation.

### Measured pose/posture baseline
The current detector-crop/global-heatmap-peak interpretation remains inadequate. At production thresholds every surviving real pose candidate classified `unknown`. The bounded threshold matrix also failed to recover enough discrimination. At the most permissive tested pair — pose confidence 0.10 and required-keypoint confidence 0.02 — only **24/53 pre-fall frames (45.3%)** became upright, **87/212 hard-negative frames (41.0%)** became upright, only **8/96 fall-window frames (8.3%)** became down, and **45/96 fall-window frames** were still unknown. Lower thresholds are therefore not promoted.

The next measured hypothesis is decoder/input semantics rather than another pose model: the pinned upstream implementation uses full-frame aspect-preserving preprocessing, local-maxima NMS and part-affinity-field grouping, while the current repository baseline isolates the detector crop, stretches it to the model input and takes one global peak per required keypoint channel.

## Evidence/data baseline
### GMDCSA-24 staged-real seed
- Primary repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`.
- Repository license at that revision: MIT. Associated Data in Brief paper: CC BY 4.0, DOI `10.5281/zenodo.13354453`.
- Positive seed: `Subject 1/Fall/05.mp4`, Git blob `4e13ed24f6c2af7062b64b1920ef706c51efe94b`, local SHA-256 `1aad4e2ee93b5498283fdc4a0478a5913dffc2ac8e2538574cd821ffb1b485f1`, annotated fall 1.8–5.0 s.
- Hard negative: `Subject 1/ADL/15.mp4`, Git blob `903da9245132cf70c10124edd0625c958f702cb8`, local SHA-256 `2cf0d421cfd8e34280bf02afc67a4f1d1abee3cd87a23f11e69ef393fedc6fdb`.
- Media remains outside public GitHub and is re-hashed around measured execution.

### Runtime/model provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Pose artifact: `human-pose-estimation-0001` FP16, already exact-size/hash reviewed.
- Evidence detector: `person-detection-0200` FP16, already exact-size/hash reviewed.
- OpenVINO Runtime: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- Reference decoder source: `demos/common/python/model_zoo/model_api/models/open_pose.py` at the same pinned OMZ commit; adapted diagnostic code preserves Intel copyright and Apache-2.0 attribution.

## Efficiency / execution ledger
One worker, one acceptance-moving work item, at most one implementation PR. Intake snapshot: `main=ebc3953b492d78a3ce0dc92f2cc2c1db7d5f5cc1`, open implementation PRs `0`, active runs for live main head `0`, unchanged retries `0`, CI-triggering requests this work session `0`, consecutive sessions without tested acceptance progress `0`. The next PR run is request **1/2**. No paid resource, self-hosted runner, home/customer media, new model family or additional worker is introduced by this work item.

## Reproduce
Repository checks:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Rights-bound evidence path in an authorized internet-connected no-spend environment:

```sh
python -m analytics_lab.validation_seed --output-dir /path/to/private-validation
python -m analytics_lab.validation_cli --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.detector_diagnostics --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.detector_thresholds \
  --manifest /path/to/private-validation/validation-manifest.json \
  --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.pose_diagnostics \
  --manifest /path/to/private-validation/validation-manifest.json \
  --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.openpose_diagnostics \
  --manifest /path/to/private-validation/validation-manifest.json \
  --candidate-dir /path/to/private-detector-cache
```

## Next executable decision
Measure the pinned reference semantics against the same positive and hard-negative clips. Record association coverage, decoded-pose counts, pre/during/after posture counts, hard-negative down classifications, pose inference/decode time and FPS. Keep this path only if it materially improves upright/down discrimination over the measured simplified baseline without introducing an unacceptable hard-negative down rate. A deterministic failure gets one root-cause correction plus regression; no blind rerun.

If the reference semantics recover useful posture discrimination, the next work item is the smallest integration-safe pose semantic correction followed by the same held-out evidence. If they do not, the reviewed pose model's decoding assumptions have been disproven on this seed and up to three rights-cleared alternative pose paths may then be evaluated. Do not alter tracking or temporal persistence until pose/posture produces discriminative real-video output.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; functional pose/posture discrimination on real video; false-alert and missed-event measurements at meaningful scale; alert-latency distribution once events fire; latency/resource envelope; platform/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Synthetic tests and this two-clip staged-real seed do not establish commercial accuracy or commercial readiness.
