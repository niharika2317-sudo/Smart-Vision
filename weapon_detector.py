from ultralytics import YOLO
import cv2
import config
from utils import box_iou


def filter_weapon_detections(detections, frame_shape, class_thresholds=None,
                             base_threshold=None, max_area_ratio=None):
    class_thresholds = (getattr(config, "WEAPON_CLASS_CONFIDENCE", {})
                        if class_thresholds is None else class_thresholds)
    base_threshold = float(config.WEAPON_CONFIDENCE if base_threshold is None else base_threshold)
    max_area_ratio = (float(config.WEAPON_MAX_BOX_AREA_RATIO) if max_area_ratio is None else float(max_area_ratio))
    frame_area = max(1, int(frame_shape[0]) * int(frame_shape[1]))
    kept = []
    for detection in detections:
        if float(detection.get("confidence", 0.0)) < float(class_thresholds.get(detection.get("label"), base_threshold)):
            continue
        x1, y1, x2, y2 = map(float, detection["box"])
        if (x2-x1) * (y2-y1) / frame_area > max_area_ratio:
            continue
        kept.append(detection)
    return kept


class WeaponPersistence:
    """Require repeated overlapping detections before a weapon enters threat scoring."""
    def __init__(self, confirm_frames=None):
        self.previous = []
        self.confirm_frames = int(confirm_frames if confirm_frames is not None else config.WEAPON_CONFIRM_FRAMES)

    def update(self, detections):
        current = []
        confirmed = []
        used = set()
        for detection in detections:
            best_index, best_iou = None, config.WEAPON_TRACK_IOU
            for index, old in enumerate(self.previous):
                score = box_iou(detection["box"], old["box"])
                if index not in used and detection.get("label") == old.get("label") and score >= best_iou:
                    best_index, best_iou = index, score
            hits = self.previous[best_index]["hits"] + 1 if best_index is not None else 1
            if best_index is not None: used.add(best_index)
            tracked = {**detection, "hits": hits}
            current.append(tracked)
            if hits >= self.confirm_frames:
                confirmed.append(detection)
        self.previous = current
        return confirmed


class WeaponDetector:
    def __init__(self, model=None):
        self.model = model or YOLO("weights/best.pt")
        self.device = config.resolve_device()

    def detect(self, frame):
        thresholds = getattr(config, "WEAPON_CLASS_CONFIDENCE", {})
        model_confidence = min([float(config.WEAPON_CONFIDENCE)] + [float(v) for v in thresholds.values()])
        results = config.run_model_with_device_fallback(lambda device: self.model.predict(
            frame,
            conf=model_confidence,
            **config.model_options(device, config.WEAPON_IMAGE_SIZE),
            verbose=False
        ))
        self.device = config.resolve_device()

        weapons = []

        if len(results) == 0:
            return weapons

        result = results[0]

        if result.boxes is None:
            return weapons

        names = self.model.names

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = names[cls]
            # Weapon center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            weapons.append({
                "label": label,
                "confidence": conf,
                "box": (x1, y1, x2, y2),
                "center": (cx, cy)
            })

        return filter_weapon_detections(weapons, frame.shape, thresholds, config.WEAPON_CONFIDENCE,
                                        config.WEAPON_MAX_BOX_AREA_RATIO)

    def draw(self, frame, weapons):
        annotated = frame.copy()
        for weapon in weapons:
            x1, y1, x2, y2 = weapon["box"]
            cx, cy = weapon["center"]
            cv2.rectangle(
                annotated,
                (x1, y1), (x2, y2),
                (0, 0, 255),
                2
            )

            # Center Point
            cv2.circle(
                annotated,
                (cx, cy),
                5,
                (255, 0, 0),
                -1
            )

            # Label
            cv2.putText(
                annotated,
                f"{weapon['label']} candidate {weapon['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return annotated
