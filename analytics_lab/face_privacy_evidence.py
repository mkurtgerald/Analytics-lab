"""Synthetic-only face privacy integration evidence; not detection accuracy."""
from __future__ import annotations
import json, time
from .face_privacy import FaceDetection, FacePrivacyConfig, apply_face_privacy
from .tracking import NormalizedBox

class _Detector:
    def detect(self, _frame):
        return (
            FaceDetection(.99, NormalizedBox(.18,.20,.42,.68)),
            FaceDetection(.97, NormalizedBox(.58,.22,.84,.70)),
        )

def run():
    import cv2, numpy as np
    frame=np.zeros((180,320,3),dtype=np.uint8)
    for y in range(180):
        for x in range(320):
            v=255 if ((x//4)+(y//4))%2 else 0
            frame[y,x]=(v,255-v,v)
    original=frame.copy()
    start=time.perf_counter()
    blurred=apply_face_privacy(frame,_Detector(),config=FacePrivacyConfig())
    denied=apply_face_privacy(frame,_Detector(),request_unblur=True,authorized_unblur=False)
    allowed=apply_face_privacy(frame,_Detector(),request_unblur=True,authorized_unblur=True)
    changed=np.any(blurred.frame_bgr!=original,axis=2)
    if not np.array_equal(frame,original): raise RuntimeError("input mutated")
    if blurred.audit.action!="blur" or denied.audit.action!="unblur_denied" or allowed.audit.action!="unblur":
        raise RuntimeError("privacy policy mismatch")
    if int(changed.sum())<=0 or np.any(changed[:20,:20]): raise RuntimeError("blur bounds mismatch")
    if not np.array_equal(allowed.frame_bgr,original): raise RuntimeError("authorized unblur mismatch")
    return {
      "evidence":"face-privacy-synthetic-integration-v1",
      "detector":"synthetic-box boundary only",
      "face_count":2,
      "default_action":blurred.audit.action,
      "unauthorized_unblur_action":denied.audit.action,
      "authorized_unblur_action":allowed.audit.action,
      "changed_pixels":int(changed.sum()),
      "input_mutated":False,
      "elapsed_seconds":round(time.perf_counter()-start,6),
      "opencv_version":cv2.__version__,
      "claim":"synthetic integration evidence only; not face-detection accuracy",
    }

def main():
    print(json.dumps(run(),sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
