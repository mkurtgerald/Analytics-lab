# Wikimedia CC0 pedestrian sequence review

## Decision

Admit one small real-video source for **Detection + Tracking evidence acquisition only**: Wikimedia Commons file `Video Codec Test pedestrian area 1080p25.y4m.webm`.

This is not a commercial-accuracy dataset and it does not arrive with exhaustive multi-object labels. Its purpose is to provide a rights-clear, static-camera, multi-person sequence from which Analytics Lab can create a tiny independent exhaustive annotation set before comparing tracker association behavior.

## Authoritative source and rights

- File page / rights record pinned for review: `https://commons.wikimedia.org/w/index.php?title=File:Video_Codec_Test_pedestrian_area_1080p25.y4m.webm&oldid=1196486240`
- Canonical media URL: `https://upload.wikimedia.org/wikipedia/commons/a/ae/Video_Codec_Test_pedestrian_area_1080p25.y4m.webm`
- Author: Taurus Media Technik
- License: CC0 1.0 Universal Public Domain Dedication (`CC0-1.0`)
- Wikimedia's reviewed file page states that copying, modification, distribution and commercial use are permitted without asking permission.
- The page records that a Wikimedia reviewer confirmed the external source was available under the stated license.

Engineering rights decision for this evidence-only path: **commercial product-development evaluation of these evidence bytes is authorized by the reviewed CC0 dedication**. This is an engineering provenance conclusion, not a legal opinion or commercial-release approval.

## Published asset identity

The reviewed Wikimedia file page records:

- length: 15.125 seconds;
- dimensions: 1920 x 1080;
- byte size: 11,215,394;
- SHA-1: `51e89a672896e45cca17aa46cd223630a6266e26`;
- description: static shot of a pedestrian area, with people passing close to the camera and high depth of field.

The first evidence run intentionally does **digest discovery only**: stream the exact canonical media bytes once on the GitHub-hosted Linux evidence runner, enforce the published byte count and SHA-1, compute SHA-256, retain no media, emit no frames, run no detector, and make no tracking-accuracy claim. The discovered SHA-256 must be pinned before any later frame extraction or annotation work.

## Why this source

This source is substantially smaller than the previously rejected/blocked fallback datasets, has unusually clear rights for commercial evaluation, uses a static camera, and contains multiple pedestrians at changing depths with close-passing/occlusion stress. That makes it appropriate for a small engineering association comparison once independent exhaustive labels exist.

It is **not** sufficient for production accuracy claims, broad robustness, vehicle/object validation, demographic coverage, low-light claims, or commercial promotion.

## Required next steps

1. Verify exact bytes and discover SHA-256 on `evidence/tracking-cc0-*`; no media retention.
2. Pin SHA-256 fail-closed in repository code.
3. Select a small fixed frame window before model inspection and extract only that bounded window ephemerally.
4. Create independent exhaustive person boxes/track labels for every measured frame; do not use the benchmarked detector/tracker outputs as ground truth.
5. Bind annotation SHA-256 and ordered frame-manifest SHA-256 through `TrackingEvidenceManifest`.
6. Run the unchanged simple-IoU baseline and portable ByteTrack slice on identical detector observations and report raw association metrics plus wall/CPU cost. Treat the result only as bounded engineering evidence.
