# Dataset review — UVify/NCSOFT human tracking

Status: reviewed candidate for Detection + Tracking evidence; payload not yet admitted.

## Source identity

- Repository: `uvify-public/human_tracking_dataset`
- Pinned revision: `eb3af0cfe49de018a0c4736581daadd8eb860883`
- Repository license file blob: `fab36c2f10dc9d2602bef9c9570df9c978598f03`
- Copyright notice: `(c) 2022 NCSOFT Corporation & UVify Co., Ltd. All Rights Reserved.`
- Published license: Creative Commons Attribution 4.0 International (`CC-BY-4.0`)
- Dataset download published by the repository through the UVify research SharePoint link.

This engineering review records the repository's published terms and provenance. It is not legal advice and does not authorize a commercial release by itself.

## Published dataset semantics

The pinned README states that the dataset contains 500 drone videos for multi-object human tracking. Extracted images are human-annotated with frame number, person ID, tracking ID, bounding box, validity, pose, occlusion, truncation and visibility. The published split reports 13,500 train images and 4,500 test images. The README reports 49,258 occluded object annotations.

For Analytics Lab evaluation, `tracking_id` is the only source identifier mapped to `GroundTruthObject.object_id`. The source `person_id` is deliberately not propagated into the platform-neutral contract because Analytics Lab track IDs are session-local association identifiers, not identity claims.

## Admission boundary

The actual image/annotation payload has not been downloaded, hashed, committed, cached or measured in this repository. The published SharePoint payload was not accessible through the current bounded execution path. Therefore:

- no media hash is asserted;
- no sequence identity is asserted;
- no detector/tracker accuracy result is asserted;
- no subject identity or biometric claim is made;
- no source payload belongs in public GitHub.

Before a sequence can become evidence, admit the smallest exact test sequence, bind exact files and cryptographic hashes, preserve the required attribution, keep the payload ephemeral, and record the image dimensions and annotation identity used for normalization.

## Adapter contract

`analytics_lab.uvify_tracking` parses the published 12-column comma-separated annotation format fail-closed. It admits only `is_valid == 1` rows, validates published flag/range semantics, rejects out-of-frame boxes rather than silently clamping them, enforces row/object ceilings, and maps only dataset `tracking_id` values into `uvify-track:<id>` ground-truth identifiers.

The adapter performs no media acquisition, detector inference, tracker inference, face recognition, re-identification or accuracy scoring. Synthetic parser tests validate the contract only and are not real-video evidence.
