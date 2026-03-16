from vision.yellow_line_detector import YellowLineDetector
from camera.camera_manager import CameraManager
import cv2

camera = CameraManager()
detector = YellowLineDetector()

print("Press Q to exit")

while True:

    ret, frame = camera.get_frame()

    if not ret:
        break

    result = detector.detect(frame)

    if result["detected"]:

        for line in result["lines"]:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1,y1), (x2,y2), (0,255,255), 3)

        print("Boundary detected:", result["position"])

    cv2.imshow("Boundary Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()