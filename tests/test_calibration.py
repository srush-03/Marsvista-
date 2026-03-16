from vision.detector_factory import DetectorFactory
from camera.camera_manager import CameraManager
import cv2

detector = DetectorFactory.create_detector()
camera = CameraManager()

print("Calibration test")
print("Move object closer and farther")
print("Press Q to exit")

while True:

    ret, frame = camera.get_frame()

    if not ret:
        break

    detections = detector.match_frame(frame)

    for d in detections:

        print(
            f"Feature: {d['feature_type']} "
            f"| score: {d['confidence']}"
        )

    cv2.imshow("Calibration Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()