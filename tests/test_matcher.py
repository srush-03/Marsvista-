from vision.detector_factory import DetectorFactory
from camera.camera_manager import CameraManager
import cv2

detector = DetectorFactory.create_detector()
camera = CameraManager()

print("Matcher test running")
print("Press Q to exit")

while True:

    ret, frame = camera.get_frame()

    if not ret:
        break

    detections = detector.match_frame(frame)

    for det in detections:

        print(
            f"Detected: {det['feature_type']} | "
            f"confidence={det['confidence']}"
        )

    cv2.imshow("Matcher Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()