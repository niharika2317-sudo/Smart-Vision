import cv2

from pose_detector import PoseDetector
from behavior_rules import BehaviorRules

pose = PoseDetector()
rules = BehaviorRules()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = pose.detect(frame)

    output = frame.copy()

    if len(results):

        result = results[0]

        if result.keypoints is not None:

            output = result.plot()

            for kp in result.keypoints.xy:

                behavior = rules.detect(kp.tolist())

                cv2.putText(
                    output,
                    behavior,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Pose Test", output)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()