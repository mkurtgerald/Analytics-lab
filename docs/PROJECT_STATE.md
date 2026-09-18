# Project state — 2026-09-17

## Product direction
Analytics Lab develops platform-independent video analytics for paid integration into the owner's products. It remains separate from K5-Vision and EdgeVMS. No production release, cross-repository write, customer/home-camera use, paid compute, GPU workload, license change, or commercial release is authorized by this repository workflow.

## Current person-down path
The person-down **candidate** path under measured validation is:

`authorized local video -> reviewed detector -> spatial continuity -> reviewed OpenPose model -> bounded pose association -> conservative posture classification -> temporal persistence -> evidence-linked candidate event -> labeled evaluation/aggregation`

Candidate events do not infer injury, cause, fault, intent, negligence or medical condition. Real-video evidence binds exact media hashes, reviewed model/runtime identities, decoded coverage, continuity, timing and evaluation metrics without retaining video in public GitHub.

## Accepted Subject-1 milestone
PR #33 was accepted and merged at `5a63abd6201a7739d7c4cd8eb95a46fdbc77e67d`. Its exact head passed Linux, Windows, the Analytics quality gate and the bounded rights-cleared real-video lane on attempt 1.

On the pinned Subject-1 pair, the retained evidence-only configuration produced the first matched staged-real person-down candidate: **1 positive episode, 1 candidate, 1 match, 0 misses, 0 hard-negative false alerts**, across about **0.0036 decoded camera-hours**. Alert delay was **4024 ms** from one matched event. The positive had 85 qualifying `down` frames. Bounded unknown-gap tolerance bridged 21 `unknown` frames, longest bridged gap 96 ms, and recovered a 3008 ms / 70-sample qualifying run while `upright`, `other`, low-confidence `down`, expiry and over-bound gaps stayed fail-closed.

This remains a tiny engineering seed, not a commercial accuracy claim.

## Held-out Subject-2 result — accepted evidence, algorithm not promoted
PR #34 was merged to live `main` at `49a558ee2df95afb990de377c1484c40892b7afc`. The unchanged exact PR head `2269196bda90da7f85d6247c02941ec7130ae442` passed Linux, **217 repository tests**, Windows, the Analytics quality gate, and the bounded real-video lane on attempt 1. Post-merge Linux/Windows/quality also passed on `main`.

The disjoint Subject-2 pair produced a legitimate generalization failure:

- positive `Subject 2/Fall/01.mp4`: **1 labeled positive, 0 candidates, 0 matches, 1 miss**;
- prone sleeping normal `Subject 2/ADL/12.mp4`: **0 candidates, 0 false alerts**;
- decoded camera-hours: **0.0038578** across the two clips;
- aggregate evidence-only end-to-end throughput on the hosted CPU run: about **63.5 FPS** for 417 frames.

The dominant positive error moved upstream to person detection. At the retained 0.10 detector floor, the labeled fall interval selected only **94/138 frames = 68.12%**, below the predeclared 85% continuity coverage bar. Link quality on selected detections remained **89.19%**, so identity continuity was not the primary deficit. The normal-negative retained **216/227 = 95.15%** detector coverage, **98.56%** link rate, and zero candidate/false alert.

Downstream evidence confirms that simply weakening temporal persistence is the wrong response. The Subject-2 positive produced 30 qualified `down` frames, but the longest qualified run was only **768 ms / 23 samples** because detector/association loss created 96 `unknown` frames. The prone sleeping negative produced 17 `down`-like frames yet its longest qualified run was only **960 ms / 17 samples** and correctly emitted no candidate. The maximum bridged unknown gap already reached **736 ms** of the existing 750 ms bound.

The prior bounded donor shootout already consumed the permitted three detector alternatives and retained `person-detection-0200`; no fourth detector search is authorized by this work item. Subject 2 remains held out from any future training/adaptation data.

## Current acceptance-moving work — same-detector orientation recovery
Live `main` at intake is `49a558ee2df95afb990de377c1484c40892b7afc`; there were **0 open implementation PRs** and no active run for that main head. The single evidence branch is `evidence/person-down-orientation-fallback`.

Before spending a training cycle, this work tests the smallest measured explanation for the current detector miss: `person-detection-0200` is upright-oriented and loses the person as the body becomes horizontal. The correction does **not** add a detector, weights, donor, training framework, data source, paid resource, or production behavior.

On a primary-orientation miss only, and only when a prior continuity box exists, the exact same pinned detector is run on bounded `numpy.rot90` views at -90 and +90 degrees. Detections are mapped back to source coordinates and only a box that clears the existing 0.10 confidence floor **and** existing 0.05 IoU continuity floor may be admitted. Primary-orientation detections always win. No prior box means no rotated reacquisition. The original OpenPose, pose-association, posture and temporal rules remain unchanged.

The same held-out Subject-2 clips are rerun for direct comparison. Retain this evidence-only correction only if it materially improves the positive detector/selection deficit without unacceptable regression. The primary comparison bar is the already-declared **85% positive fall-window coverage**; hard-negative candidate events and false alerts must remain **0**. Record fallback attempts/accepts, mapped/linked candidates, selection coverage by labeled window, matched/missed events, false alerts, alert delay, posture/temporal trace and CPU cost. If orientation recovery does not clear a meaningful acceptance improvement, do not broaden it; the next safe path is a rights-pinned, subject-separated detector adaptation/training experiment.

## Evidence/data provenance
### GMDCSA-24
- Primary repository: `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos`.
- Pinned revision: `5abac7693229900cf80f722e878fbb119211fc1c`, authored 2024-08-21 and aligned with v2.1.
- Repository license at that revision: MIT.
- Versioned dataset provenance: Zenodo v2.1 DOI `10.5281/zenodo.13354453`; associated paper DOI `10.1016/j.dib.2024.110892`.
- Subject-1 positive SHA-256: `1aad4e2ee93b5498283fdc4a0478a5913dffc2ac8e2538574cd821ffb1b485f1`.
- Subject-1 hard negative SHA-256: `2cf0d421cfd8e34280bf02afc67a4f1d1abee3cd87a23f11e69ef393fedc6fdb`.
- Held-out Subject-2 positive SHA-256: `0448c122dbfdffc423cc6e282cb54f60d801fc1c741de9aafca3f051db7c1abb`.
- Held-out Subject-2 prone-normal SHA-256: `24559694cceadbcf1bb34f217967c233e412b97e25f9dd9be3f733f4e16067df`.
- Media remains ephemeral/outside public GitHub and is re-hashed before and after measured execution.

### Runtime/model provenance
- Open Model Zoo commit: `6697dead54ed1cdd664b0313189c2cb52ee6335e`, Apache-2.0.
- Pose artifact: `human-pose-estimation-0001` FP16, exact size/hash reviewed.
- Evidence detector: `person-detection-0200` FP16, exact size/hash reviewed.
- OpenVINO Runtime: `2026.3.1`; evidence decoder: `opencv-python-headless==4.12.0.88`.
- Reference OpenPose decoder source is pinned to the same OMZ commit with attribution preserved.

## Efficiency ledger
One worker, one acceptance-moving work item, at most one implementation PR. No additional model family, training job, paid resource, self-hosted runner, home/customer media or duplicate agent is introduced.

At this work-item intake: open implementation PRs **0**, active runs for live main **0**, unchanged retries **0**, CI-triggering requests **0/2**, consecutive sessions without tested acceptance progress **0**. The execution environment available to the worker does not provide a local clone/runtime, so a local machine preflight is not claimed; live repository/head/PR/run counts were tool-verified before mutation. Branch preparation is batched before opening the one PR so exact-head CI/evidence runs once.

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
python -m analytics_lab.detector_thresholds --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.pose_diagnostics --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.openpose_diagnostics_v2 --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.person_down_e2e_diagnostics --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
python -m analytics_lab.person_down_orientation_diagnostics --manifest /path/to/private-validation/validation-manifest.json --candidate-dir /path/to/private-detector-cache
```

## Next executable decision
Run exact-head CI plus the existing bounded Subject-2 real-video lane once. Compare the same-detector orientation result directly to the accepted 68.12% positive fall-window detector coverage and zero-negative-alert baseline. Keep the change only for measured acceptance improvement without hard-negative regression. Do not tune temporal persistence, add a fourth detector, or admit Subject 2 into training.

Merge only when exact-head Linux, Windows and Analytics quality are green on the unchanged tested base and the bounded real-video evidence lane completes. Main post-merge verification intentionally does not reacquire real-video/model evidence.

## Outstanding commercial-release gates
Expanded held-out positive/negative real-video evidence across genuinely different cameras/sites and multi-person scenes; generalizable detector recall/identity behavior; false-alert and missed-event measurements at meaningful scale; alert-latency distribution; latency/resource envelope; privacy/security/provenance review; platform/native dependency provenance; versioned installable integration adapter; packaging/notices; and explicit owner release approval. Small GMDCSA-24 subject rotations do not establish commercial accuracy or commercial readiness.
