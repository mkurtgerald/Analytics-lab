# Public Webcam Data Policy

Public webcams are a preferred source of real-world validation footage for Analytics Lab, alongside permissively licensed datasets, synthetic/test clips, and controlled local camera footage.

## Purpose
Use public webcams to expose analytics to realistic scene diversity: weather, lighting, perspective, crowd density, traffic, camera vibration, compression artifacts, occlusion, and long-duration background behavior.

## Rules
1. A stream being publicly viewable does not automatically mean its footage may be downloaded, retained, redistributed, or used for model training.
2. Every source must have a provenance record before use beyond transient live testing.
3. Prefer sources with explicit public-use, open-data, research, or redistribution terms.
4. For sources without explicit reuse rights, limit use to transient evaluation where legally and contractually appropriate; do not archive or redistribute footage by default.
5. Do not use public webcams for face recognition, identity inference, demographic profiling, or persistent person re-identification.
6. Minimize retention of raw footage. Prefer derived telemetry, event metadata, anonymized annotations, or short clips only when rights allow.
7. Respect source terms of service, robots/access restrictions, rate limits, jurisdictional privacy law, and takedown requests.
8. Do not bypass authentication, paywalls, stream protections, or access controls.

## Source Registry
Each webcam source should record:
- Source name
- Operator/owner
- Public URL
- Stream URL if legitimately exposed
- Geographic category (roadway, beach, transit, public square, etc.)
- License / terms URL
- Permitted uses
- Retention allowed: yes/no/unknown
- Redistribution allowed: yes/no/unknown
- Training allowed: yes/no/unknown
- Last verified date
- Notes

## Preferred Uses
- Detector robustness validation
- Tracking stability tests
- Temporal event logic validation
- Scene-change and camera-health analytics
- Crowd-flow and object-state experiments
- Near-miss and traffic interaction research where lawful
- Environmental variation testing

## Not Default Training Data
Public webcams should not be assumed to be training data. Training use requires explicit confirmation that the footage's license or operator terms permit that use.

## Output Discipline
For every experiment, distinguish:
- **Observation**: directly supported by frames or metadata
- **Inference**: analytic interpretation
- **Ground truth**: independently labeled or verified reference

This separation is required for defensible benchmarking and future production use.
