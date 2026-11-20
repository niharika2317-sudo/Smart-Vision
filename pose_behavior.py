from pose_detector import PoseDetector
from behavior_rules import BehaviorRules
import config
import time


class PoseBehavior:
    def __init__(self, detector=None):
        self.pose = detector or PoseDetector()
        self.rules = BehaviorRules()
        self.last_inference_seconds = 0.0

    def detect(self, frame):
        started = time.perf_counter()
        results = self.pose.detect(frame)
        self.last_inference_seconds = time.perf_counter() - started
        if not results:
            return []
        result = results[0]
        if result.keypoints is None or result.boxes is None:
            return []
        boxes = result.boxes.xyxy.tolist()
        keypoints = result.keypoints.xy.tolist()
        confidences = result.keypoints.conf.tolist() if result.keypoints.conf is not None else None
        poses = []
        for i, (box, kp) in enumerate(zip(boxes, keypoints)):
            confidence = confidences[i] if confidences else None
            confident_count = (sum(float(value) >= config.MEDICAL_POSE_KEYPOINT_CONFIDENCE for value in confidence)
                               if confidence is not None else 0)
            poses.append({"box": tuple(map(float, box)),
                          "behavior": self.rules.detect(kp, confidence),
                          "confident_keypoints": confident_count,
                          "keypoint_confidences": [float(value) for value in confidence] if confidence else []})
        return poses
