# Road-vehicle donor engineering review — 2026-10-20

The existing OMZ 0106 barrier detector emitted no proposals on a public Caltrans
traffic still containing four or five provisionally annotated distant vehicles.
That measured domain gap justifies testing one road-scene donor before training.

Select Intel `person-vehicle-bike-detection-crossroad-0078` from
[`openvinotoolkit/open_model_zoo@6697dead54ed1cdd664b0313189c2cb52ee6335e`](https://github.com/openvinotoolkit/open_model_zoo/tree/6697dead54ed1cdd664b0313189c2cb52ee6335e/models/intel/person-vehicle-bike-detection-crossroad-0078).
The pinned model.yml explicitly links the Apache-2.0 repository license.
Exact FP16 XML/BIN sizes, SHA-384 identities and official URLs are retained in
`analytics_lab.road_vehicle` and the accompanying measurement record.
The two files total 2,883,302 bytes; no weights enter the repository.

The publisher describes surveillance scenes and a Crossroad Camera demo.
Input is BGR NCHW 1x3x1024x1024; verified IR declares float32. Preserve raw
0–255 values with linear resizing, without invented external mean/scaling.
Output is SSD 1x1x200x7: class 1 person, class 2 vehicle, class 3 bike.
The optional adapter admits only vehicle class 2, preserves its provenance,
verifies local artifacts before loading, and performs no network acquisition.
Use the existing OpenVINO 2026.3.1/OpenCV 4.12.0.88/NumPy 2.2.6 versions.

The published vehicle AP is 74.94% on the publisher's validation set; it is
neither this camera's score nor a guarantee of 90% performance. Training corpus
names, independent overlap and dataset redistribution rights are unspecified.
Model engineering admission does not grant access to that training corpus.
No training data, model redistribution, production release or default change
is admitted. The frozen one-image comparison remains retrospective and uses
unresolved provisional AI annotations, without human-gold qualification.

Primary README, model.yml and license blob identities are recorded in
`ops/reviews/public-webcam-accuracy-20261020.json`. The camera's public-material
rights are reviewed separately from the donor model license. Real held-out
video with exhaustive boxes, normal negatives and appropriate plate resolution
is still required to qualify detection, tracking or end-to-end OCR.
