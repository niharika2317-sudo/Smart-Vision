from behavior_detector import BehaviorDetector
from pose_behavior import PoseBehavior
from behavior_fusion import BehaviorFusion
from utils import box_iou, point_in_box
import config
import time


class BehaviorPipeline:
    def __init__(self, pose_behavior=None):
        self.motion = BehaviorDetector()
        self.pose = pose_behavior or PoseBehavior()
        self.fusion = BehaviorFusion()
        self.last_pose_seconds = 0.0
        self.last_logic_seconds = 0.0

    def detect(self, frame, persons, weapons, t=0.0):
        if not persons:
            self.last_pose_seconds = 0.0
            self.last_logic_seconds = 0.0
            return []
        poses = self.pose.detect(frame)
        self.last_pose_seconds = getattr(self.pose, "last_inference_seconds", 0.0)
        logic_started = time.perf_counter()
        used = set()
        final = []
        for person in persons:
            best_idx, best_iou = None, config.POSE_MIN_IOU
            for i, candidate in enumerate(poses):
                score = box_iou(person["box"], candidate["box"])
                if i not in used and score >= best_iou:
                    best_idx, best_iou = i, score
            pose = "Unknown"
            confident_keypoints = 0
            keypoint_confidences = []
            if best_idx is not None:
                used.add(best_idx)
                pose = poses[best_idx]["behavior"]
                confident_keypoints = int(poses[best_idx].get("confident_keypoints", 0))
                keypoint_confidences = list(poses[best_idx].get("keypoint_confidences", []))
            person["pose_behavior"] = pose
            person["pose_confident_keypoints"] = confident_keypoints
            person["pose_keypoint_confidences"] = keypoint_confidences
            movement, speed = self.motion.analyze(person.get("id", -1), person["box"], t)
            person["speed"] = speed
            holding = any(point_in_box(w["center"], person["box"], config.HOLD_MARGIN) for w in weapons)
            label = self.fusion.fuse(movement, pose, holding)
            person["behavior_label"] = label
            final.append(label)
        self.last_logic_seconds = time.perf_counter() - logic_started
        return final
