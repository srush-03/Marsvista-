import time
import cv2
from camera.camera_manager import CameraManager

camera = CameraManager()

frame_count = 0
start = time.time()

print("Running FPS test for 30 seconds")

while True:

    ret, frame = camera.get_frame()

    if not ret:
        break

    frame_count += 1

    elapsed = time.time() - start

    cv2.imshow("FPS Test", frame)

    if elapsed > 30:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

fps = frame_count / elapsed

print("\n======================")
print("FPS RESULTS")
print("======================")

print("Frames processed:", frame_count)
print("Elapsed time:", round(elapsed,2))
print("FPS:", round(fps,2))

camera.release()
cv2.destroyAllWindows()