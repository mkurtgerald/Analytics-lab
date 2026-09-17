# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current product path
The implemented person-down **candidate** path is:

`authorized local video -> reviewed detector + pose model -> temporary IoU tracking -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> held-out evaluation/aggregation -> rights-bound validation evidence`

Candidate events do not infer injury, cause, fault, or intent. Validation binds exact local-media SHA-256 identity, reviewed model-artifact identities, runtime/device identity, decoded coverage, throughput and evaluation metrics without retaining video. Media is re-hashed after execution so changed media cannot inherit stale evidence.

## Current acceptance-moving work
PR #26 / `evidence/0200-threshold-sensitivity` is the single implementation vehicle, based on green `main` revision `ee318e19963d95f5488c8b4d84b9b09ddfaa0770`. It follows merged PR #25, which exhausted the permitted three-detector comparison and found no promotable replacement at confidence 0.50.

Exact-head run `35270446841` on revision `e51df5c95f8ce3e8605066e81cb78a5327ba3b0c` passed the repository guardrail preflight, 181 Linux repository tests (one expected optional-OpenCV skip before runtime installation), synthetic replay, rights-bound real-video validation and detector diagnostics, the new threshold-sensitivity evidence command, Windows regression and the final Analytics quality gate on attempt 1. Runtime evidence used OpenVINO `2026.3.1-22476-759c5a6ab8c-releases/2026/3`, OpenCV `4.12.0`, NumPy `2.2.6`, the exact two GMDCSA-24 clips, and the exact hash-pinned `person-detection-0200` artifacts. No media/model workflow artifacts were uploaded.

### Measured detector threshold result
`person-detection-0200` was run once per frame and the same raw outputs were evaluated at confidence thresholds 0.50, 0.40, 0.30, 0.20 and 0.10. The predeclared evidence bars were >=55% positive fall-interval coverage, >=90% hard-negative person coverage and <=5% frames with multiple person detections on each clip. The measured result was `recommended_threshold: null`.

- **0.50:** fall-interval coverage 29/96 = **30.2%**; positive duplicate-frame rate **19.0%**; hard-negative coverage **100%**; negative duplicate-frame rate **6.6%**.
- **0.40:** fall-interval coverage 40/96 = **41.7%**; positive duplicate-frame rate **32.2%**; hard-negative coverage **100%**; negative duplicate-frame rate **9.9%**.
- **0.30:** fall-interval coverage 61/96 = **63.5%**; positive duplicate-frame rate **38.5%**; hard-negative coverage **100%**; negative duplicate-frame rate **11.3%**.
- **0.20:** fall-interval coverage 82/96 = **85.4%**; positive duplicate-frame rate **44.3%**; hard-negative coverage **100%**; negative duplicate-frame rate **12.7%**.
- **0.10:** fall-interval coverage 96/96 = **100%**; positive duplicate-frame rate **80.5%**; hard-negative coverage **100%**; negative duplicate-frame rate **19.3%**.

This is a useful negative result. The prone/fall signal is present below confidence 0.50: lowering the threshold materially recovers the missed fall frames. But every tested threshold violates the declared multiple-detection/noise bound, and the noise burden increases sharply as recall improves. Therefore threshold lowering alone is rejected as the detector fix. No fourth detector will be added and no unchanged threshold rerun is justified.

The threshold diagnostic measured detector-only throughput at about **134.4 FPS** in this hosted run. That number is retained as a run-local resource observation only; hosted-runner hardware variance means it is not used as a cross-run promotion claim.

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
One worker, one acceptance-moving work item, one implementation PR. Intake verification: `main` at `ee318e19963d95f5488c8b4d84b9b09ddfaa0770`; its post-merge Analytics quality run `35262537924` passed on attempt 1; open implementation PRs were 0 and active runs for the live main head were 0 before this PR. PR #26 was CI-triggering request 1/2 and produced new measured evidence on attempt 1; unchanged retries remain 0. This evidence-state update is the second and final CI-triggering mutation permitted for this session. No extra worker, framework, detector, dataset, runner, paid resource, home/customer footage, training run or licensing shortcut was introduced.

The local offline preflight command could not be executed before the first mutation in the automation container because that container could not resolve external hosts to materialize the public repository files. The exact PR workflow subsequently ran the repository's own guardrail preflight successfully against base `ee318e19963d95f5488c8b4d84b9b09ddfaa0770` and head `e51df5c95f8ce3e8605066e81cb78a5327ba3b0c`; no guardrail was waived.

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
Finish exact-head verification for PR #26 after this measured evidence-state update and merge only if Linux real-video evidence, Windows regression and the final Analytics quality gate remain green on the unchanged tested base.

Then close threshold tuning. The evidence now shows that `person-detection-0200` contains sub-threshold prone-person signal, but confidence lowering cannot recover it within the declared multiple-detection/noise bound. Define the smallest rights-cleared **detector adaptation/fine-tuning experiment** that specifically raises confidence/quality on prone and transitioning people while preserving normal-person precision. Use only the already reviewed commercially eligible sources and no-spend resources; keep a held-out portion of real GMDCSA-24 video out of training and continue to treat synthetic UE4 data as synthetic augmentation only. Do not launch a large training job until the exact training code, base-weight rights, train/holdout split, resource ceiling and acceptance delta are pinned.

Once detector recall is materially improved, attack the already-proven pose/posture failure (`unknown` on every real candidate) before changing tracking or temporal persistence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across cameras/sites; materially improved detector recall; functional pose/posture discrimination on real video; false-alert and missed-event measurements at meaningful scale; alert-latency distribution once events fire; latency/resource envelope; platform/native dependency provenance; security/privacy/provenance review; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Synthetic/stub tests and this two-clip staged-real seed do not establish commercial accuracy or commercial readiness.
