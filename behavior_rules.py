import math
import config


class BehaviorRules:
    """Pose rules use relative geometry so they work across image sizes."""
    def detect(self, keypoints, conf=None):
        if len(keypoints) < 17:
            return "Unknown"
        pts = [tuple(map(float, p[:2])) for p in keypoints]
        confidence = list(conf) if conf is not None else [1.0] * len(pts)
        def good(i):
            return i < len(confidence) and confidence[i] >= config.KEYPOINT_CONFIDENCE and pts[i] != (0.0, 0.0)
        if not all(good(i) for i in (5, 6, 11, 12)):
            return "Unknown"
        sx, sy = (pts[5][0] + pts[6][0]) / 2, (pts[5][1] + pts[6][1]) / 2
        hx, hy = (pts[11][0] + pts[12][0]) / 2, (pts[11][1] + pts[12][1]) / 2
        angle = math.degrees(math.atan2(abs(hx - sx), abs(hy - sy)))
        if angle > config.FALL_ANGLE_DEGREES:
            return "Possible Fall"
        shoulder_width = max(1.0, math.hypot(pts[5][0] - pts[6][0], pts[5][1] - pts[6][1]))
        for wrist, shoulder in ((9, 5), (10, 6)):
            if good(wrist) and pts[wrist][1] < pts[shoulder][1] - config.RAISED_HAND_WIDTH_RATIO * shoulder_width:
                return "Raised Hand"
        return "Normal"
