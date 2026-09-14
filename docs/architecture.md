# Analytics Lab Architecture

## Purpose

Analytics Lab is an independent R&D environment for developing video analytics that may later be consumed by K5 Vision or other properly licensed platforms.

## Design principles

1. **Analytics are modules, not VMS features.** The lab should avoid hard dependencies on a specific UI, recorder, database, or deployment model.
2. **Primitive-first architecture.** Detection, tracking, pose, segmentation, OCR, temporal state, and scene reasoning should be reusable across multiple analytics.
3. **Standard event contract.** Every analytic emits a common event envelope regardless of its internal model stack.
4. **Replaceable donor components.** External detectors and libraries should sit behind adapters so they can be upgraded or replaced without rewriting the analytic.
5. **Evidence-aware outputs.** An analytic should be able to express uncertainty, supporting observations, and abstention rather than forcing a binary alert.
6. **Temporal reasoning.** The project should support event chains over time, not just frame-by-frame classifications.

## Logical layers

```text
Video / RTSP / File Input
        |
        v
Perception Primitives
  - object detection
  - pose estimation
  - segmentation
  - OCR / text
  - optical flow
        |
        v
Tracking + Scene State
  - persistent objects
  - zones
  - trajectories
  - object state changes
        |
        v
Temporal Event Engine
  - event windows
  - sequence rules
  - correlation
  - confidence fusion
        |
        v
Analytics
  - slip/fall causality
  - distress/man-down
  - near miss
  - behavioral escalation
  - scene integrity
  - blind-spot risk
  - incident genesis
        |
        v
Standard Event Contract
        |
        +--> Benchmark / Evaluation
        +--> K5 adapter
        +--> Other licensed integrations
```

## Initial package boundaries

- `core/detection` — detector interfaces and adapters
- `core/tracking` — identities and trajectories
- `core/pose` — skeletal/keypoint interfaces
- `core/segmentation` — masks and scene regions
- `core/temporal` — event windows and state machines
- `core/scene_state` — persistent representation of scene conditions
- `core/event_engine` — correlation, confidence fusion, and event production
- `analytics` — analytic-specific logic
- `integrations` — video ingestion and downstream adapters
- `benchmarks` — metrics, fixtures, and reproducible evaluations
- `datasets` — metadata and acquisition instructions, not unlicensed media

## Integration boundary

K5 Vision should consume Analytics Lab through a versioned event interface rather than importing experimental internal code directly. This keeps the lab independently testable and lets K5 promote only mature analytics.
