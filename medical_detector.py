import cv2
import numpy as np
import config


def _intersection_over_person(box, other_box):
    x1, y1, x2, y2 = map(float, box)
    ox1, oy1, ox2, oy2 = map(float, other_box)
    intersection = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(0.0, min(y2, oy2) - max(y1, oy1))
    area = max(1.0, (x2 - x1) * (y2 - y1))
    return intersection / area


class MedicalDetector:
    """Conservative medical heuristics; blood color is supporting evidence only."""
    def __init__(self):
        self.state = {}
        self.diagnostics = []

    def detect(self, frame, persons, poses, t):
        cases, present = [], set()
        self.diagnostics = []
        frame_height, frame_width = frame.shape[:2]
        edge_x = frame_width * config.MEDICAL_EDGE_MARGIN_RATIO
        edge_y = frame_height * config.MEDICAL_EDGE_MARGIN_RATIO

        for person, pose in zip(persons, poses):
            track = int(person.get("id", -1))
            if track < 0:
                continue
            present.add(track)
            x1, y1, x2, y2 = map(float, person["box"])
            width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
            aspect = width / max(1.0, height)
            height_ratio = height / max(1.0, frame_height)
            touches_edge = (x1 <= edge_x or y1 <= edge_y or
                            x2 >= frame_width - edge_x or y2 >= frame_height - edge_y)
            overlap_fractions = [_intersection_over_person(person["box"], other["box"])
                                 for other in persons if other is not person and other.get("box")]
            max_overlap = max(overlap_fractions, default=0.0)
            overlap_count = sum(value > 0 for value in overlap_fractions)

            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(frame_width, int(x2)), min(frame_height, int(y2))
            roi = frame[iy1:iy2, ix1:ix2]
            blood_ratio = float(person["blood_ratio"]) if "blood_ratio" in person else 0.0
            if "blood_ratio" not in person and roi.size:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                red = (((hsv[:, :, 0] <= config.BLOOD_HUE_LOW) | (hsv[:, :, 0] >= config.BLOOD_HUE_HIGH))
                       & (hsv[:, :, 1] >= config.BLOOD_SAT_MIN)
                       & (hsv[:, :, 2] >= config.BLOOD_VALUE_MIN)
                       & (hsv[:, :, 2] <= config.BLOOD_VALUE_MAX))
                blood_ratio = float(red.mean())
            possible_blood = config.MEDICAL_BLOOD_MIN_RATIO <= blood_ratio <= config.MEDICAL_BLOOD_MAX_RATIO

            state = self.state.setdefault(track, {"down": None, "blood": None, "first_seen": t, "seen": t})
            state["seen"] = t
            track_age = max(0.0, t - state["first_seen"])
            track_old_enough = track_age >= config.MEDICAL_MIN_TRACK_AGE_SECONDS
            box_large_enough = height_ratio >= config.MEDICAL_MIN_BOX_HEIGHT_RATIO
            not_heavily_occluded = max_overlap < config.MEDICAL_MAX_OCCLUSION_RATIO
            geometry_ok = (track_old_enough and box_large_enough and not touches_edge and not_heavily_occluded)

            confident_keypoints = int(person.get("pose_confident_keypoints", 0))
            pose_label = person.get("pose_behavior", pose)
            pose_fall = (pose == "Possible Fall" or pose_label == "Possible Fall")
            pose_fall = pose_fall and confident_keypoints >= config.MEDICAL_POSE_FALL_MIN_KEYPOINTS
            aspect_fall = aspect >= config.LYING_ASPECT
            try:
                speed = abs(float(person.get("speed", float("inf"))))
            except (TypeError, ValueError):
                speed = float("inf")
            stationary = speed <= config.MEDICAL_STATIONARY_SPEED
            down = geometry_ok and stationary and (pose_fall or aspect_fall)

            state["down"] = (state["down"] if state["down"] is not None else t) if down else None
            state["blood"] = (state["blood"] if state["blood"] is not None else t) if possible_blood and geometry_ok else None
            down_confirmed = state["down"] is not None and t - state["down"] >= config.MEDICAL_FALL_SECONDS
            blood_confirmed = state["blood"] is not None and t - state["blood"] >= config.MEDICAL_BLOOD_SECONDS
            reasons = []
            if down_confirmed:
                reasons.append("Person Down")
            # Blood color can support an already confirmed down-person case, never alert alone.
            if down_confirmed and blood_confirmed:
                reasons.append("Possible Blood")

            self.diagnostics.append({
                "track_id": track, "box": list(map(int, person["box"])),
                "box_width_px": int(width), "box_height_px": int(height),
                "box_height_fraction": round(height_ratio, 4), "aspect_ratio": round(aspect, 4),
                "touches_frame_edge": touches_edge, "overlapping_people_count": overlap_count,
                "max_overlap_fraction": round(max_overlap, 4), "track_age_seconds": round(track_age, 3),
                "pose_label": pose_label, "pose_fall_signal": pose_fall,
                "confident_keypoints": confident_keypoints, "aspect_fall_signal": aspect_fall,
                "stationary": stationary, "speed": None if speed == float("inf") else speed,
                "blood_ratio": round(blood_ratio, 5), "blood_heuristic_signal": possible_blood,
                "geometry_eligible": geometry_ok, "reported_reasons": reasons,
            })
            if reasons:
                cases.append({"track_id": track, "severity": "HIGH" if len(reasons) == 2 else "MEDIUM",
                              "reasons": reasons, "box": person["box"]})

        for track in list(self.state):
            if track not in present and t - self.state[track]["seen"] > config.TRACK_FORGET_SECONDS:
                del self.state[track]
        return cases
