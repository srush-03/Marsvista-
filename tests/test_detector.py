from vision.detector_factory import DetectorFactory
from camera.camera_manager import CameraManager
import cv2

detector = DetectorFactory.create_detector()
camera = CameraManager()

print("Press Q to exit")

while True:

    ret, frame = camera.get_frame()
    if not ret:
        break

    detections = detector.match_frame(frame)

    for d in detections:

        x, y, w, h = d["bbox"]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(
            frame,
            d["feature_type"],
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

    cv2.imshow("Detector Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()