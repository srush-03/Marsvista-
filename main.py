import cv2
import os
import time
import json
import numpy as np
from datetime import datetime

from cnn_seed_loader import CNNSeedLoader
from cnn_matcher     import CNNMatcher
from yellow_line_detector import YellowLineDetector
from metadata_logger import MetadataLogger
from instance_tracker import InstanceTracker

# ── CONFIG ────────────────────────────────────────────────────────
SEEDS_FOLDER      = "seeds"
KEYFRAMES_FOLDER  = "keyframes"
DETECTIONS_FOLDER = "detections"
LOGS_FOLDER       = "logs"

CAMERA_INDEX      = 0
USE_OAK_D         = False        # True when OAK-D Lite connected

CNN_THRESHOLD     = 0.65         # lowered from 0.75 — consecutive filter handles FP
KEYFRAME_FPS      = 8            # save rate for stitching
KEYFRAME_INTERVAL = 1.0 / KEYFRAME_FPS   # 0.125s

# Quality filters for keyframe saving
MIN_BRIGHTNESS    = 30
MAX_BRIGHTNESS    = 220
MIN_SHARPNESS     = 50.0

# Instance tracker
PIXEL_DIST        = 120
TIME_WINDOW       = 4.0
# ─────────────────────────────────────────────────────────────────


def open_camera():
    if USE_OAK_D:
        try:
            import depthai as dai
            pipeline = dai.Pipeline()
            cam      = pipeline.create(dai.node.ColorCamera)
            cam.setResolution(
                dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setFps(30)
            cam.setInterleaved(False)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            xout = pipeline.create(dai.node.XLinkOut)
            xout.setStreamName("rgb")
            cam.preview.link(xout.input)
            device = dai.Device(pipeline)
            queue  = device.getOutputQueue(
                name="rgb", maxSize=4, blocking=False)
            print("OAK-D Lite opened.")
            return "oakd", device, queue
        except Exception as e:
            print(f"OAK-D failed ({e}), using webcam...")

    for idx in range(5):
        cap = cv2.VideoCapture(idx)   # no CAP_DSHOW — cross platform
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"Webcam opened at index {idx}")
            return "webcam", cap, None

    return None, None, None


def read_frame(cam_type, cap_or_device, queue):
    if cam_type == "oakd":
        pkt = queue.get()
        return True, pkt.getCvFrame()
    return cap_or_device.read()


def frame_quality_ok(frame):
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    sharpness  = cv2.Laplacian(gray, cv2.CV_64F).var()
    return (MIN_BRIGHTNESS < brightness < MAX_BRIGHTNESS
            and sharpness > MIN_SHARPNESS)


# ── SETUP ─────────────────────────────────────────────────────────
for folder in [KEYFRAMES_FOLDER, LOGS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

print("=" * 55)
print("  ISRO IRoC-U 2026 — Perception System")
print("=" * 55)

loader  = CNNSeedLoader(SEEDS_FOLDER)
for ftype in loader.get_vectors():
    os.makedirs(os.path.join(DETECTIONS_FOLDER, ftype), exist_ok=True)

matcher        = CNNMatcher(loader, threshold=CNN_THRESHOLD)
yellow_detector = YellowLineDetector()
tracker        = InstanceTracker(PIXEL_DIST, TIME_WINDOW)
logger         = MetadataLogger(LOGS_FOLDER)

cam_type, cap_or_device, queue = open_camera()
if cam_type is None:
    print("ERROR: No camera")
    exit()

print("\nSystem ready. Press Q to quit.\n")

# ── STATE ─────────────────────────────────────────────────────────
frame_index       = 0
kf_count          = 0
kf_skipped        = 0
last_kf_time      = 0.0

kf_meta = {"fps_target": KEYFRAME_FPS, "frames": []}
kf_meta_path = os.path.join(KEYFRAMES_FOLDER, "capture_metadata.json")

# ── MAIN LOOP ─────────────────────────────────────────────────────
while True:
    ret, frame = read_frame(cam_type, cap_or_device, queue)
    if not ret or frame is None:
        break

    frame_index += 1
    now         = time.time()
    display     = frame.copy()

    # ══════════════════════════════════════════════════════════════
    # TASK A: KEYFRAME SAVING AT 8 FPS
    # ══════════════════════════════════════════════════════════════
    if now - last_kf_time >= KEYFRAME_INTERVAL:
        last_kf_time = now

        if frame_quality_ok(frame):
            kf_filename = f"kf_{kf_count:06d}.jpg"
            kf_path     = os.path.join(KEYFRAMES_FOLDER, kf_filename)
            cv2.imwrite(kf_path, frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kf_meta["frames"].append({
                "filename"   : kf_filename,
                "frame_index": frame_index,
                "timestamp"  : round(now, 3),
                "brightness" : round(float(gray.mean()), 1),
                "sharpness"  : round(float(
                    cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
            })
            with open(kf_meta_path, 'w') as f:
                json.dump(kf_meta, f, indent=2)

            kf_count += 1
        else:
            kf_skipped += 1

    # ══════════════════════════════════════════════════════════════
    # TASK B: YELLOW LINE DETECTION (every frame)
    # ══════════════════════════════════════════════════════════════
    yellow_result = yellow_detector.detect(frame)
    if yellow_result['detected']:
        warning = yellow_result['direction_warning']
        if yellow_result['lines']:
            for line in yellow_result['lines']:
                x1, y1, x2, y2 = line[0]
                cv2.line(display, (x1,y1), (x2,y2), (0,255,255), 3)
        cv2.putText(display,
            f"YELLOW LINE — {warning}",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2)
        print(f"[YELLOW] fr{frame_index:06d} | {warning}")

    # ══════════════════════════════════════════════════════════════
    # TASK C: CNN DETECTION (async — non-blocking)
    # ══════════════════════════════════════════════════════════════
    matcher.submit_frame(frame)
    scores     = matcher.get_all_scores()
    detections = matcher.match_frame()
    hits       = matcher.get_consecutive_hits()

    # Score bars
    y_pos = 70
    for ftype, score in scores.items():
        streak = hits.get(ftype, 0)
        color  = (0,255,0) if score >= CNN_THRESHOLD else (0,0,255)
        cv2.rectangle(display,
            (10, y_pos-18), (10+int(score*280), y_pos+2), color, -1)
        cv2.putText(display,
            f"{ftype}: {score:.3f}  [{streak}/3]",
            (15, y_pos-3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        y_pos += 32

    for det in detections:
        ftype      = det['feature_type']
        confidence = det['confidence']
        x, y, pw, ph = det['bbox']
        center     = det['bbox_center']

        instance_id, is_new = tracker.update(
            ftype, center, confidence, frame_index)

        # Save detection images
        det_fname = f"det_{ftype}_{instance_id}_fr{frame_index:06d}.jpg"
        ctx_fname = f"ctx_{ftype}_{instance_id}_fr{frame_index:06d}.jpg"
        det_path  = os.path.join(DETECTIONS_FOLDER, ftype, det_fname)
        ctx_path  = os.path.join(DETECTIONS_FOLDER, ftype, ctx_fname)

        patch = frame[y:y+ph, x:x+pw]
        if patch.size > 0:
            cv2.imwrite(det_path, patch)
        cv2.imwrite(ctx_path, frame)

        logger.log_detection(
            frame_index     = frame_index,
            feature_type    = ftype,
            detection       = det,
            image_path      = det_path,
            full_frame_path = ctx_path,
            drone_pose      = {"x": 0, "y": 0, "z": 4.5},
            instance_id     = instance_id,
            is_new_instance = is_new
        )

        box_color = (0,255,0) if is_new else (0,200,255)
        cv2.rectangle(display, (x,y), (x+pw,y+ph), box_color, 3)
        cv2.putText(display,
            f"{instance_id} {'★NEW' if is_new else 're-seen'} {confidence:.2f}",
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        if is_new:
            print(f"[NEW] {instance_id} | fr{frame_index:06d} | {confidence:.3f}")

    # ── HUD ───────────────────────────────────────────────────────
    h = display.shape[0]
    confirmed = tracker.count_confirmed()
    cv2.putText(display,
        f"fr:{frame_index}  kf:{kf_count}@{KEYFRAME_FPS}fps  skip:{kf_skipped}",
        (10, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(display,
        "Confirmed: " + (
            "  ".join([f"{k}:{v}" for k,v in confirmed.items()])
            if confirmed else "scanning..."
        ),
        (10, h-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(display,
        "Raw hits: " + "  ".join(
            [f"{k}:{v}" for k,v in logger.detection_counts.items()]
        ),
        (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    cv2.imshow("ISRO Perception System", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── SHUTDOWN ──────────────────────────────────────────────────────
matcher.stop()
if cam_type == "webcam":
    cap_or_device.release()
cv2.destroyAllWindows()
logger.log_instance_summary(tracker.get_summary())

print(f"\n{'='*55}")
print(f"  FLIGHT COMPLETE")
print(f"{'='*55}")
print(f"  Keyframes saved  : {kf_count} @ {KEYFRAME_FPS}fps")
print(f"  Keyframes skipped: {kf_skipped} (quality rejected)")
print(f"  Confirmed instances:")
for ftype, count in tracker.count_confirmed().items():
    print(f"    {ftype}: {count}")
print(f"  Logs      : {LOGS_FOLDER}/")
print(f"  Detections: {DETECTIONS_FOLDER}/")
print(f"{'='*55}")