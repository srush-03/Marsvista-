import cv2
import time
from cnn_seed_loader import CNNSeedLoader
from cnn_matcher import CNNMatcher

print("Loading CNN models...")
loader  = CNNSeedLoader("seeds")
matcher = CNNMatcher(loader, threshold=0.75)
print("Models loaded.\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fps_list      = []
frame_index   = 0
test_duration = 30
start_time    = time.time()
prev_time     = time.time()

print("Running 30 second FPS test...")
print("Point camera at your seed object — watch streak counter go 1→2→3 then CONFIRMED\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1

    # ── MAIN THREAD: submit frame + read results (non-blocking) ──
    matcher.submit_frame(frame)           # sends frame to bg thread
    scores     = matcher.get_all_scores() # reads last computed scores
    detections = matcher.match_frame()    # reads last confirmed detections
    hits       = matcher.get_consecutive_hits()

    # ── FPS ───────────────────────────────────────────────────────
    now  = time.time()
    fps  = 1.0 / (now - prev_time + 1e-9)
    fps_list.append(fps)
    prev_time    = now
    recent_fps   = sum(fps_list[-10:]) / len(fps_list[-10:])

    display = frame.copy()

    color = (0,255,0) if recent_fps >= 15 else \
            (0,165,255) if recent_fps >= 8 else (0,0,255)
    cv2.putText(display, f"FPS: {recent_fps:.1f}",
        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.putText(display,
        f"Frame {frame_index}  |  Detection thread running in background",
        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    # Score bars + streak counter
    y = 115
    for ftype, score in scores.items():
        streak = hits.get(ftype, 0)
        c = (0,255,0) if score >= 0.75 else (0,0,255)
        cv2.rectangle(display, (10,y-18),(10+int(score*300),y+2), c, -1)
        cv2.putText(display,
            f"{ftype}: {score:.3f}   streak: {streak}/3",
            (15, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += 38

    # Detection banner
    if detections:
        for det in detections:
            x, yb, pw, ph = det['bbox']
            cv2.rectangle(display, (x,yb),(x+pw,yb+ph),(0,255,0),3)
            cv2.putText(display,
                f"CONFIRMED: {det['feature_type']} {det['confidence']:.2f}",
                (10, display.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    else:
        cv2.putText(display, "scanning...",
            (10, display.shape[0]-20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    elapsed = now - start_time
    cv2.putText(display,
        f"Test ends: {max(0,test_duration-elapsed):.0f}s",
        (display.shape[1]-200, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

    cv2.imshow("FPS Test (Async)", display)

    if elapsed >= test_duration:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

matcher.stop()
cap.release()
cv2.destroyAllWindows()

if fps_list:
    stable = sum(fps_list[5:]) / len(fps_list[5:]) if len(fps_list) > 5 else fps_list[0]
    print("\n" + "="*50)
    print("  FPS RESULTS (ASYNC)")
    print("="*50)
    print(f"  Frames processed : {frame_index}")
    print(f"  Stable FPS       : {stable:.1f}")
    print(f"  ms per frame     : {1000/stable:.0f}ms")
    print("="*50)
    if stable >= 20:
        print("  ✅ EXCELLENT — Camera runs at full speed")
    elif stable >= 15:
        print("  ✅ GOOD — Ready for flight")
    elif stable >= 8:
        print("  ⚠️  OKAY — Workable, monitor on Jetson")
    else:
        print("  ❌ Still slow — tell me, we'll reduce crop count")
    print("="*50)