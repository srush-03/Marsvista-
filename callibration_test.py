import cv2
import time
import numpy as np
from cnn_seed_loader import CNNSeedLoader
from cnn_matcher import CNNMatcher

print("Loading models...")
loader  = CNNSeedLoader("seeds")
matcher = CNNMatcher(loader, threshold=0.50)  # low threshold to see all scores
print("Ready.\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Score history for each feature type — track max seen
max_scores   = {ft: 0.0 for ft in loader.get_vectors()}
score_history = {ft: [] for ft in loader.get_vectors()}

frame_index = 0
start = time.time()

print("Instructions on screen. Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    matcher.submit_frame(frame)
    scores = matcher.get_all_scores()
    hits   = matcher.get_consecutive_hits()

    # Track max scores seen
    for ft, sc in scores.items():
        if sc > max_scores[ft]:
            max_scores[ft] = sc
        score_history[ft].append(sc)
        if len(score_history[ft]) > 30:
            score_history[ft].pop(0)

    display = frame.copy()
    h, w    = display.shape[:2]

    # Instructions
    cv2.putText(display,
        "CALIBRATION: Move object close → far → away",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Score display — large and clear
    y = 80
    for ft, sc in scores.items():
        # Rolling average over last 10 frames
        hist = score_history[ft]
        avg  = sum(hist[-10:]) / len(hist[-10:]) if hist else 0

        # Color: green=good, orange=borderline, red=low
        if sc >= 0.70:
            color = (0, 255, 0)
            label = "STRONG"
        elif sc >= 0.60:
            color = (0, 200, 255)
            label = "MEDIUM"
        elif sc >= 0.50:
            color = (0, 165, 255)
            label = "WEAK"
        else:
            color = (0, 0, 255)
            label = "LOW"

        # Big score bar
        bar_w = int(sc * 500)
        cv2.rectangle(display, (10, y), (10 + bar_w, y + 40), color, -1)
        cv2.rectangle(display, (10, y), (510, y + 40), (100,100,100), 2)

        cv2.putText(display,
            f"{ft}:  {sc:.3f}  [{label}]  avg={avg:.3f}  max={max_scores[ft]:.3f}",
            (15, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        cv2.putText(display,
            f"{ft}:  {sc:.3f}  [{label}]  avg={avg:.3f}  max={max_scores[ft]:.3f}",
            (15, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        y += 60

    # Threshold guide lines on bar
    cv2.line(display, (int(0.65*500)+10, 75), (int(0.65*500)+10, y),
             (0, 255, 255), 2)
    cv2.putText(display, "0.65", (int(0.65*500)+5, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.line(display, (int(0.75*500)+10, 75), (int(0.75*500)+10, y),
             (255, 255, 0), 2)
    cv2.putText(display, "0.75", (int(0.75*500)+5, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    # Bottom instructions
    cv2.putText(display,
        "Step 1: Hold object CLOSE  |  Step 2: 1m away  |  Step 3: 2m away  |  Step 4: No object",
        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    cv2.putText(display,
        f"Note down scores at each step. Press Q when done.",
        (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Calibration Test", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

matcher.stop()
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*55)
print("  CALIBRATION RESULTS")
print("="*55)
for ft in loader.get_vectors():
    hist = score_history[ft]
    if hist:
        print(f"\n  {ft}:")
        print(f"    Max score seen   : {max_scores[ft]:.3f}")
        print(f"    Final avg score  : {sum(hist[-10:])/len(hist[-10:]):.3f}")
print("\n  NEXT STEP:")
print("  Tell me:")
print("  1. Score when object is CLOSE (like seed image angle)")
print("  2. Score when object is 1-2m away")
print("  3. Score when pointing at random background")
print("  I will set the exact right threshold for your setup.")
print("="*55)