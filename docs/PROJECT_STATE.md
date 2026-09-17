# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed model-artifact identities, runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video. Media is re-hashed after execution so changed media cannot inherit stale evidence.

## Current acceptance-moving work
`evidence/0200-threshold-sensitivity` is the single implementation vehicle, based on green `main` revision `ee318e19963d95f5488c8b4d84b9b09ddfaa0770`. It follows merged PR #25, which exhausted the permitted three-detector comparison and found no promotable replacement at confidence 0.50.

The measured detector baseline remains:
- `person-detection-retail-0013`: fall-interval coverage 21/96 = **21.9%**, hard-negative coverage 100%, detector-only about 136 FPS.
- `person-detection-0200`: 29/96 = **30.2%**, hard-negative coverage 100%, about 295 FPS. It is the only reviewed alternative that improved prone/fall coverage, but the +8.3 percentage-point gain missed the predeclared +25-point promotion bar.
- `person-detection-0202`: 14/96 = **14.6%**, a regression.
- `person-detection-0203`: zero detections on the two-clip seed under the bounded common adapter at 0.50.

No fourth detector is permitted. The current work therefore changes approach rather than expanding donor search: run `person-detection-0200` once per frame and derive fixed thresholds 0.50, 0.40, 0.30, 0.20 and 0.10 from the same raw outputs. For each threshold record positive fall-interval coverage, hard-negative coverage, total detections, duplicate frames/excess detections and CPU detector throughput. A threshold is evidence-eligible only if fall-interval coverage reaches at least 55%, hard-negative person coverage remains at least 90%, and duplicate-frame rate stays at or below 5% on both clips. The highest threshold clearing all bars is preferred; otherwise the result is `null` and threshold tuning is rejected as the detector solution.

The evidence workflow now replaces the exhausted four-model shootout with this single-model threshold diagnostic, reducing repeated model acquisition/inference while preserving the same hosted Linux five-minute ceiling, read-only GitHub permissions, exact media/model checks and non-retention boundary.

The full person-down candidate still produces zero events, one missed positive episode and zero false alerts on this deliberately tiny staged-real seed. This is measured engineering evidence, not a commercial accuracy claim. The independently measured pose/posture defect also remains: every real pose candidate classified `unknown`. Tracking remains downstream of detector and pose/posture failures.

## Evidence/data baseline
### GMDCSA-24 — bounded staged-real validation source
- Primary repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`.
- Repository license at that revision: MIT. Associated Data in Brief paper: CC BY 4.0, DOI `10.5281/zenodo.13354453`, describing 81 fall and 79 ADL clips from four subjects in three home setups and use for fall-detection training/testing.
- Positive seed: `Subject 1/Fall/05.mp4`, Git blob `4e13ed24f6c2af7062b64b1920ef706c51efe94b`, 5,872,655 bytes, local SHA-256 `1aad4e2ee93b5498283fdc4a0478a5913dffc2ac8e2538574cd821ffb1b485f1`, annotated fall 1.8-5.0 s.
- Hard negative: `Subject 1/ADL/15.mp4`, Git blob `903da9245132cf70c10124edd0625c958f702cb8`, 7,233,079 bytes, local SHA-256 `2cf0d421cfd8e34280bf02afc67a4f1d1abee3cd87a23f11e69ef393fedc6fdb`.
- Exact media identity is verified before and after measured execution; media remains outside public GitHub.

### Other reviewed sources
- UE4 Fall Detection Dataset pinned at `55041766dea68eaddc1df1c06aabd0a51931a22a`, CC BY 4.0, synthetic only; eligible for commercial training/evaluation with attribution but never real-video accuracy evidence.
- Figshare article `28596332` version 2, posted 2025-03-14, CC BY 4.0, real-world secondary source; the ~2.36 GB corpus remains outside the current bounded-resource path.
- UR Fall remains excluded because its official source states non-commercial terms. Roboflow mirrors with unclear upstream provenance remain hold-only.

## Runtime / detector provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Current baseline artifacts: `person-detection-retail-0013` FP16 + `human-pose-estimation-0001` FP16, exact size/SHA-384 verified.
- Threshold candidate `person-detection-0200` FP16: XML 254,619 bytes / SHA-384 `654a515935f6dffc0440cffdeeb6a889bc25bf5d64e425ce2337070cdbcce1b1fbbcf42e4b2761265eb65ec07c1cd6f7`; BIN 3,634,654 / `10b6f79b495ad1ef13748938c452379c6b6cd825d11426ffa68be3eeed6c48a0af04e0b913c843e6e7431aa1c13f5471`.
- OpenVINO Runtime baseline: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- No shootout detector is integrated into the production-candidate backend.

## Efficiency / execution ledger
One worker, one acceptance-moving work item, at most one implementation PR. Intake verification: `main` at `ee318e19963d95f5488c8b4d84b9b09ddfaa0770`; its post-merge Analytics quality run `35262537924` passed on attempt 1; open implementation PRs 0; active runs for the live main head 0; unchanged retries 0; CI-triggering requests in this session 0; no-progress sessions 0 because PR #25 produced measured detector evidence. The proposed PR counts as 1/1 WIP. No extra worker, framework, detector, dataset, runner, paid resource, home/customer footage, training run or licensing shortcut is added.

The local offline preflight command could not be executed in this automation environment because the container cannot resolve external hosts to materialize the public repository files. Live counts and policy were nevertheless read directly through the connected GitHub repository before mutation; this limitation does not waive any guardrail or permit an extra PR/CI request.

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
```

## Next executable step
Open one evidence PR and execute this exact bounded threshold diagnostic once. If no threshold clears the declared recall/negative/noise bars, stop detector threshold tuning and use the accumulated measured donor evidence to define the smallest rights-cleared fine-tuning/training experiment within the no-spend envelope. If a threshold clears all bars, integrate it only as an evidence candidate and rerun the complete person-down path before any promotion.

Once detector recall is materially improved, attack the already-proven pose/posture failure (`unknown` on every real candidate) before changing tracking or temporal persistence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across cameras/sites; materially improved detector recall; functional pose/posture discrimination on real video; false-alert and missed-event measurements at meaningful scale; alert-latency distribution once events fire; latency/resource envelope; platform/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Synthetic/stub tests and this two-clip staged-real seed do not establish commercial accuracy or commercial readiness.
