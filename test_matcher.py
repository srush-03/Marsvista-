import cv2
import os

# Quick seed capture tool
os.makedirs("seeds/feature_type_1", exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Position the cube in front of camera")
print("Press S to save as seed image")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(display, f"Press S to save seed | Saved: {count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Seed Capture", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        path = f"seeds/feature_type_1/seed{count+1}.jpg"
        cv2.imwrite(path, frame)
        count += 1
        print(f"Saved: {path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done. {count} seed images saved.")
