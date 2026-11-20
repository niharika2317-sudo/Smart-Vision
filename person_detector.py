from ultralytics import YOLO
import cv2
import config


class PersonDetector:
    def __init__(self, model=None):
        self.model = model or YOLO("yolov8n.pt")
        self.device = config.resolve_device()

    def reset(self):
        predictor=getattr(self.model,"predictor",None)
        for tracker in getattr(predictor,"trackers",[]) or []:
            reset=getattr(tracker,"reset",None)
            if reset: reset()

    def detect(self, frame):

        results = config.run_model_with_device_fallback(lambda device: self.model.track(
            frame,
            persist=True,
            tracker=config.PERSON_TRACKER,
            classes=[0],
            conf=config.PERSON_CONFIDENCE,
            **config.model_options(device, config.PERSON_IMAGE_SIZE),
            verbose=False
        ))
        self.device = config.resolve_device()

        annotated = results[0].plot()

        boxes = results[0].boxes

        persons = []

        person_count = 0

        if boxes is not None:

            person_count = len(boxes)

            for box in boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                track_id = -1

                if box.id is not None:
                    track_id = int(box.id[0])

                persons.append({
                    "id": track_id,
                    "center": (cx, cy),
                    "box": (x1, y1, x2, y2)
                })

                cv2.circle(
                    annotated,
                    (cx, cy),
                    5,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    annotated,
                    f"ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

        cv2.putText(
            annotated,
            f"Persons : {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return annotated, person_count, persons
