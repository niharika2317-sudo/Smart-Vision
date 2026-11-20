from collections import defaultdict, deque
import config
from utils import center


class BehaviorDetector:
    """Classify net centre displacement in body-heights per second."""
    def __init__(self):
        self.history = defaultdict(deque)
        self.last_seen = {}

    def analyze(self, track_id, box, t):
        if track_id is None or int(track_id) < 0:
            return "Unknown", 0.0
        track_id = int(track_id)
        for old_id, seen in list(self.last_seen.items()):
            if t - seen > config.TRACK_FORGET_SECONDS:
                self.history.pop(old_id, None)
                self.last_seen.pop(old_id, None)
        self.last_seen[track_id] = t
        height = max(1.0, abs(float(box[3]) - float(box[1])))
        points = self.history[track_id]
        points.append((float(t), center(box), height))
        while len(points) > 2 and t - points[0][0] > config.MOTION_WINDOW_SECONDS:
            points.popleft()
        if len(points) < 2 or points[-1][0] <= points[0][0]:
            return "Unknown", 0.0
        elapsed = points[-1][0] - points[0][0]
        median_height = sorted(p[2] for p in points)[len(points) // 2]
        height = max(1.0, median_height)
        net_distance = ((points[-1][1][0] - points[0][1][0]) ** 2 +
                        (points[-1][1][1] - points[0][1][1]) ** 2) ** 0.5
        # Net displacement misses a runner who changes direction or runs in place.
        # Ignore small per-sample box jitter before adding the tracked path length.
        jitter = float(getattr(config, "MOTION_JITTER_RATIO", 0.12)) * height
        path_distance = 0.0
        for previous, current in zip(points, list(points)[1:]):
            step = ((current[1][0] - previous[1][0]) ** 2 +
                    (current[1][1] - previous[1][1]) ** 2) ** 0.5
            path_distance += max(0.0, step - jitter)
        net_speed = net_distance / elapsed / height
        path_speed = path_distance / elapsed / height
        speed = max(net_speed, 0.75 * path_speed)
        if speed >= config.AGGRESSIVE_SPEED: label = "Aggressive Movement"
        elif speed >= config.RUNNING_SPEED: label = "Running"
        elif speed >= config.WALKING_SPEED: label = "Walking"
        else: label = "Standing"
        return label, speed
