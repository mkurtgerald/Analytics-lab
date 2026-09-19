# Lightweight OpenPose training-lineage review — 2026-09-18

## Decision

**Admit the upstream trainer lineage only as a fail-closed adaptation candidate. Training remains blocked.**

This review does not download a checkpoint, train a model, change the retained runtime model, or authorize commercial release. The current evidence still supports `person-detection-0200 -> OMZ human-pose-estimation-0001 -> corrected OpenPose decoding -> bounded association -> posture -> 3000 ms persistence` as the retained path while the pose-representation bottleneck is investigated.

## Exact trainer lineage

- Repository: `Daniil-Osokin/lightweight-human-pose-estimation.pytorch`
- Pinned revision: `d23c284b09acf27a163e1febd511e7482cac25ed`
- Repository license at that revision: Apache-2.0
- Upstream stated environment: Ubuntu 16.04 / Python 3.6
- Direct dependency compatibility baseline, pinned to the minimum versions declared by upstream:
  - `torch==0.4.1`
  - `torchvision==0.2.1`
  - `pycocotools==2.0.0`
  - `opencv-python==3.4.0.14`
  - `numpy==1.14.0`

The repository license clears the trainer source code itself for commercial use subject to Apache-2.0 obligations. It does **not** by itself prove commercial/redistribution rights for every binary dependency, pretrained checkpoint, training image, or derived model.

## Pretrained checkpoint hold

Upstream identifies this starting checkpoint:

`https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth`

The reviewed trainer README identifies it as the COCO-pretrained checkpoint and reports the intended conversion path. The storage index currently exposes the file, but the reviewed primary sources do not publish a cryptographic hash or an artifact-specific license/notice that Analytics Lab can bind fail-closed.

Therefore:
- exact checkpoint filename/source: pinned;
- cryptographic identity: **missing**;
- artifact-specific commercial/redistribution right: **unresolved**;
- training-data provenance for that exact checkpoint: described as COCO, but not accepted here as a substitute for artifact rights;
- status: **HOLD — do not download or train from this checkpoint yet**.

No license inference is made from the fact that the trainer repository and Open Model Zoo repository are Apache-2.0.

## Dependency-rights hold

The direct Python package versions are now explicit, but a complete transitive dependency/binary-license inventory for this legacy stack has not been established. That is intentionally a blocking condition in `analytics_lab.pose_adaptation_readiness`.

The historical environment is also old enough that execution on a current hosted runner may require compatibility work. Do not solve that by silently relaxing versions or adding another framework. Any modernization must preserve model semantics and receive its own bounded compatibility check.

## Export/runtime compatibility

The pinned trainer documents:
1. `scripts/convert_to_onnx.py --checkpoint-path <CHECKPOINT>` -> `human-pose-estimation.onnx`;
2. OpenVINO conversion with outputs `stage_1_output_0_pafs` and `stage_1_output_1_heatmaps`.

The repository also points to Open Model Zoo `human-pose-estimation-0001`, matching the retained model family. Analytics Lab targets OpenVINO Runtime `2026.3.1` CPU, but the exact pinned trainer revision has **not** been smoke-tested through a modern ONNX -> OpenVINO 2026.3.1 export in this project. That remains a hard gate before training is considered ready.

## Subject separation

No adaptation corpus is admitted yet. Train and validation subject sets therefore remain empty by construction.

The following subjects are locked as evaluation holdouts and must not enter training or validation merely to make prior misses pass:
- GMDCSA24 Subjects 2, 3 and 4;
- Figshare SBJ_01, SBJ_10, SBJ_06, SBJ_03, SBJ_02 and SBJ_09 from the three already-measured broader-source pairs.

Any future adaptation corpus must define disjoint train, validation and untouched holdout subjects before a training command can be authorized.

## First allowed resource envelope

If and only if all rights/data/export gates clear, the first adaptation smoke experiment is bounded to:
- CPU only;
- at most 2 CPU threads;
- at most 20 wall-clock minutes;
- at most 4096 MiB RAM;
- at most 2048 MiB temporary storage;
- exactly one trial;
- no paid compute and no GPU.

This is a compatibility/learning-signal smoke ceiling, not a promise that a commercially useful model can be trained within it. A later resource increase requires evidence that the bounded trial is technically justified and still remains within repository policy.

## Machine-enforced readiness blockers

`analytics_lab.pose_adaptation_readiness` now blocks training while any of the following remain unresolved:
- transitive dependency rights;
- pretrained checkpoint hash;
- pretrained checkpoint commercial/redistribution license;
- nonempty subject-separated training and validation sets;
- overlap with protected holdout subjects;
- exact OpenVINO 2026.3.1 export smoke compatibility;
- CPU/no-spend resource policy.

## Next executable step

Do **not** train yet. Resolve the checkpoint identity/license from an authoritative upstream source, or reject that checkpoint and choose a rights-cleared initialization path. In parallel, admit a rights-cleared adaptation corpus with explicit subject-separated train/validation/holdout identities. Only after those gates and the export smoke test clear may one bounded CPU adaptation trial run. Existing Subjects 2-4 and all already-measured Figshare subjects stay untouched evaluation evidence.
