from ultralytics import YOLO
import config


class PoseDetector:

    def __init__(self, model=None):
        self.model = model or YOLO("yolov8n-pose.pt")
        self.device = config.resolve_device()

    def detect(self, frame):

        results = config.run_model_with_device_fallback(lambda device: self.model.predict(
            frame,
            conf=config.POSE_DETECT_CONFIDENCE,
            **config.model_options(device, config.POSE_IMAGE_SIZE),
            verbose=False
        ))
        self.device = config.resolve_device()

        return results
