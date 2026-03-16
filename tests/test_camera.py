from camera.camera_manager import CameraManager
import cv2

camera = CameraManager(camera_type="oakd", width=1280, height=720, fps=30)

print("Press Q to exit")

while True:
    ret, frame = camera.get_frame()

    if not ret:
        print("Camera read failed")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()