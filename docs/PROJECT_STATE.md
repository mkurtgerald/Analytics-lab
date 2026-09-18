# Project state — 2026-09-18

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Retained person-down candidate path
`authorized local video -> reviewed detector -> spatial continuity/orientation recovery -> reviewed pose -> bounded pose association -> conservative posture -> temporal persistence -> evidence-linked candidate -> labeled evaluation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Media remains outside public GitHub and is re-hashed before measured execution.

## Measured real-video frontier
- **Subject 1:** first matched engineering seed in PR #33: 1 positive, 1 candidate, 1 match, 0 misses, 0 hard-negative false alerts, about 0.0036 decoded camera-hours, 4024 ms alert delay. Engineering evidence only.
- **Subject 2:** same-detector orientation recovery raised labeled-fall detector coverage to 126/138 = 91.30%. A bounded near-diagonal ambiguity correction improved the longest qualified down run to 2544 ms while preserving 3000 ms persistence and the 750 ms unknown-gap budget. The remaining interruption is genuinely upright, so Subject 2 is not tuned further.
- **Subject 3:** positive remained a miss while its sitting-to-sleeping-on-floor negative stayed at 0 false alerts. A selected-person crop recovered more poses but regressed longest qualified persistence 1184 ms -> 512 ms and increased CPU cost; PR #38 was closed unmerged.
- **Subject 4 retained path:** 0 candidates / 1 miss on the positive and 0 candidates / 0 false alerts on the dynamic prone/push-up negative across 11,632 ms / 0.0032311111 camera-hours. The labeled positive window had 129/129 detector coverage and 129/129 safe pose association, 55 qualified down observations, and a 1312 ms longest qualified run. This localizes the dominant wall to pose/posture quality rather than detector recall or association.

Exact Subject-4 media identities:
- positive `Subject 4/Fall/03.mp4` SHA-256 `4f178dea77bf9abb9036bcd7fedfbe345358fc2d63e1704dc49b7ec1a621e2f7`;
- negative `Subject 4/ADL/07.mp4` SHA-256 `6e410abbc6d4ef7bf52ac835b952a74a94311d0977fdfddb21885477602b7b69`.

After three consecutive sessions without tested acceptance improvement on the existing OpenPose/posture path, further holdout-specific threshold tuning stopped.

## Rejected alternative — OMZ 0005
PR #40 merged the bounded rights/resource pin for Open Model Zoo `human-pose-estimation-0005` FP16 at exact OMZ commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0, with the Apache-2.0 Associative Embedding reference decoder and bounded evidence dependency `scipy==1.17.1`.

PR #41 then completed a valid exact-head same-video A/B on Subject 4 after fixing the demonstrated AE grouping defect (`diff.shape[0]` / `diff.shape[1]`) and replacing its regression with a dependency-free AST contract test. Exact head `fe4aef5f5f1e82c54d14d3e105544f095d4ff283` passed Linux, 43 guardrail tests, 227 synthetic tests (1 optional OpenCV skip), Windows, the Analytics quality gate, and the real-video CPU lane on first attempt.

`0005` failed its predeclared retention rule and PR #41 was closed unmerged. Against the same-run retained `0001` baseline:
- baseline: 0 positive candidates / 1 miss, 0 negative false alerts, 55 positive qualified-down frames, 1312 ms longest qualified run, 59.43 FPS, 2803.63 ms pose inference, 513.88 ms decode, 5872.29 ms total elapsed;
- `0005`: 0 positive candidates / 1 miss, 0 negative false alerts, only 1 positive qualified-down frame, 0 ms single-sample longest run, 43.88 FPS, 4102.27 ms pose inference, 1150.83 ms decode, 7953.33 ms total elapsed;
- during the labeled positive window `0005` produced 1 `down`, 9 `other`, 117 `unknown`, 2 `upright`; the dominant failure was `required_keypoint_confidence_below_threshold` on 100 labeled-window frames under unchanged posture semantics.

No confidence threshold, posture rule, association bound, 3000 ms persistence rule or 750 ms unknown-gap rule was weakened to rescue `0005`.

## Active acceptance item — bounded OMZ 0006 resolution step
The next smallest executable pose experiment is Open Model Zoo `human-pose-estimation-0006`, not another model-family search. At the same pinned OMZ commit, `0005`, `0006` and `0007` use the **same exact FP16 weight blob**; `0006` changes the reviewed static EfficientHRNet input from 288x288 to 352x352 and published COCO AP from 45.6% to 51.1% while donor-reported compute rises from 5.9206 to 8.844 GFLOPs. The unchanged weight identity makes this a bounded resolution-step test of whether the 0005 low-keypoint-confidence failure is recoverable without changing downstream semantics.

Pinned `0006` artifacts from exact `models/intel/human-pose-estimation-0006/model.yml`:
- FP16 XML: 1,064,076 bytes, SHA-384 `567d2f921e5af960384fddc30bfe8d7110d90222dae50ce36a13d6a0ca5113ff83b9adfe7c505a403d9f46d17285da39`;
- FP16 BIN: 19,039,904 bytes, SHA-384 `ef4ab20cd0695a4b86789607acc6eb636d07fbb6786f30300ea713bb94a9110ed1ecec5db5577398ddc13663cc2ce690` — identical to the reviewed `0005` FP16 weight blob;
- license: Apache-2.0 via the pinned OMZ repository license;
- input: `1x3x352x352`; outputs: heatmaps `1x17x176x176`, embeddings `1x17x176x176x1`;
- decoder: the same pinned Apache-2.0 Associative Embedding adaptation already exercised by PR #41;
- runtime: OpenVINO 2026.3.1 CPU, OpenCV headless 4.12.0.88, SciPy 1.17.1 only in the bounded evidence lane.

### Predeclared 0006 retention rule
Run the exact same Subject-4 positive/push-up-negative pair and retained detector/orientation/association/posture/temporal/evaluator path. Change only the pose network input artifact from `0001` baseline to pinned `0006` candidate plus its already-reviewed AE decoder. Retain only if the candidate materially improves the positive acceptance target while the push-up negative remains 0 candidate events / 0 false alerts and CPU cost remains acceptable. Green CI, donor AP, or additional keypoints alone are not acceptance.

Do not lower required-keypoint confidence, weaken posture semantics, bridge genuine `upright`/`other`, widen unsafe association, reduce 3000 ms persistence, extend the 750 ms unknown-gap budget, train, add a fourth detector, or use Subjects 2–4 for training.

## Evidence/data provenance
GMDCSA-24 primary repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`, pinned revision `5abac7693229900cf80f722e878fbb119211fc1c`, MIT; Zenodo v2.1 DOI `10.5281/zenodo.13354453`, paper DOI `10.1016/j.dib.2024.110892`. Media remains ephemeral/outside public GitHub.

Runtime/model provenance: OMZ commit `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0; retained pose `human-pose-estimation-0001` FP16; evidence detector `person-detection-0200` FP16; OpenVINO Runtime 2026.3.1; headless OpenCV 4.12.0.88. Associative Embedding code retains Intel copyright/Apache attribution.

## Efficiency ledger
One worker, one acceptance-moving item, at most one implementation/evidence PR. PR #41 is closed unmerged after a decisive measured rejection. This fresh bounded continuation uses the second of at most three alternative pose configurations and the same evidence pair. No paid/GPU/self-hosted resource, home/customer media, training, production promotion, release action, or commercial accuracy claim is authorized.

The current automation work session has already used one CI-triggering PR run for the final `0005` result; opening the `0006` evidence PR is the second and final CI-triggering action permitted this session. No unchanged retry is available beyond the policy's one diagnosed-transient allowance, and no third mutation/dispatch may be used in this session.

## Reproduce
Repository checks:

```sh
python tools/guardrails.py ci
python -m unittest discover -s tests -v
python -m analytics_lab --input examples/person_down.jsonl --source-id synthetic-camera --session-id fixture-001
```

Rights-bound evidence path on an authorized GitHub-hosted Linux PR runner:

```sh
python -m analytics_lab.validation_seed --output-dir /path/to/private-validation
python -m analytics_lab.validation_cli --manifest /path/to/private-validation/validation-manifest.json
python -m analytics_lab.detector_thresholds --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.person_down_e2e_diagnostics --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.person_down_pose_ab_0006 --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
```

## Commercial-release gates still open
Substantially broader held-out positive/negative real-video evidence across genuinely different subjects/cameras/sites and multi-person scenes; meaningful false-alert and miss measurements; alert-delay distribution; latency/resource envelope; privacy/security/provenance review; native/platform dependency review; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Small GMDCSA-24 subject rotations and bounded donor A/Bs are engineering evidence only, not commercial accuracy.
