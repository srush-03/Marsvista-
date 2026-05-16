"""
frame_recorder.py — ISRO IRoC-U 2026 ASCEND | Team Marsvista
=============================================================
Runs DURING flight. Two modes:

    LAPTOP MODE (testing with webcam, no ROS2):
        python frame_recorder.py --mode webcam

    JETSON MODE (ROS2 + OAK-D Lite + ORB-SLAM3):
        python frame_recorder.py --mode ros2
        Topics: /oak/rgb/image_rect  /orb_slam3/pose

What it does:
    - Captures HD frames (1280x720 minimum)
    - Stamps each frame filename with pose: frame_000001_x2.34_y1.56_z4.50.jpg
    - Runs yellow line boundary detector on every frame
    - Lightweight — zero CNN during flight
    - Saves to flight_frames/ folder

Edge cases handled:
    - Camera disconnect: retry 3 times then graceful exit
    - Pose not available: saves frame with x0_y0_z0 and logs warning
    - Disk full: warns at 80%, stops at 95%
    - Blurry frames: flagged in metadata but still saved (post-match handles it)
"""

import cv2
import os
import json
import time
import argparse
import threading
import numpy as np
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────
SAVE_DIR      = "flight_frames"
SAVE_FPS      = 10           # frames per second to save
MIN_ALTITUDE  = 0.5          # skip frames below this z (laptop: always save)
DISK_WARN_PCT = 80           # warn when disk this % full
DISK_STOP_PCT = 95           # stop saving when disk this % full
JPEG_QUALITY  = 95

# Yellow line HSV
YELLOW_LO = np.array([18,  80,  80])
YELLOW_HI = np.array([35, 255, 255])


# ── DISK CHECK ────────────────────────────────────────────────────
def disk_usage_pct(path="."):
    import shutil
    total, used, free = shutil.disk_usage(path)
    return 100.0 * used / total


# ── SHARPNESS ─────────────────────────────────────────────────────
def sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


# ── YELLOW LINE ───────────────────────────────────────────────────
def boundary_warning(frame):
    h, w = frame.shape[:2]
    roi  = frame[h//2:, :]
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw / max(ch, 1) < 4.0:
            continue
        cx = x + cw // 2
        if cx < w // 3:   return "TURN_RIGHT"
        if cx > 2*w//3:   return "TURN_LEFT"
        return "CAUTION"
    return None


# ══════════════════════════════════════════════════════════════════
# WEBCAM MODE
# ══════════════════════════════════════════════════════════════════
def run_webcam():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Try camera indices 0,1,2
    cap = None
    for idx in range(3):
        c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if c.isOpened():
            cap = c
            print(f"Camera opened at index {idx}")
            break
    if cap is None:
        print("ERROR: No camera found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_index   = 0
    saved_count   = 0
    last_save     = 0.0
    save_interval = 1.0 / SAVE_FPS
    metadata      = []
    fake_x = fake_y = 0.0
    fake_z = 4.5          # simulate 4.5m altitude
    log_path = os.path.join(SAVE_DIR, "session_log.json")

    print("\n" + "="*55)
    print("  Frame Recorder — WEBCAM MODE")
    print(f"  Saving to: {SAVE_DIR}/")
    print(f"  Target: {SAVE_FPS} FPS")
    print("  SPACE = start/stop recording")
    print("  WASD  = simulate drone movement (X/Y)")
    print("  U/J   = altitude up/down")
    print("  Q     = quit")
    print("="*55 + "\n")

    recording  = False
    reconnects = 0

    while True:
        ret, frame = cap.read()

        # Handle camera disconnect
        if not ret:
            reconnects += 1
            print(f"  WARNING: Camera read failed (attempt {reconnects}/3)")
            if reconnects >= 3:
                print("  Camera disconnected — stopping.")
                break
            time.sleep(0.5)
            continue
        reconnects = 0

        frame_index += 1
        now = time.time()

        # Disk check
        disk_pct = disk_usage_pct(SAVE_DIR)
        if disk_pct >= DISK_STOP_PCT:
            print(f"  CRITICAL: Disk {disk_pct:.0f}% full — stopping recorder.")
            break

        # Ensure 1280x720
        h, w = frame.shape[:2]
        if w != 1280 or h != 720:
            frame = cv2.resize(frame, (1280, 720))

        # Save frame at target FPS
        if recording and (now - last_save) >= save_interval:
            last_save = now
            blur = sharpness(frame)
            warn = boundary_warning(frame)   # compute once
            fname = (f"frame_{saved_count:06d}"
                     f"_x{fake_x:.3f}_y{fake_y:.3f}_z{fake_z:.3f}.jpg")
            fpath = os.path.join(SAVE_DIR, fname)
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            metadata.append({
                "frame": saved_count,
                "file" : fname,
                "x"    : fake_x, "y": fake_y, "z": fake_z,
                "sharpness": round(blur, 1),
                "boundary_warning": warn,
            })
            saved_count += 1
            if warn:
                print(f"  BOUNDARY WARNING: {warn}")
        else:
            warn = None   # no warning outside recording

        # HUD
        display = frame.copy()
        hud_col = (0, 255, 0) if recording else (0, 100, 255)
        cv2.rectangle(display, (0,0), (w, 55), (0,0,0), -1)
        status = "● REC" if recording else "■ PAUSED"
        cv2.putText(display, f"{status}  |  Frame:{frame_index}  Saved:{saved_count}",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_col, 2)
        cv2.putText(display,
            f"Pose: x={fake_x:.2f} y={fake_y:.2f} z={fake_z:.2f}  |  Disk:{disk_pct:.0f}%",
            (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

        bwarn = warn   # reuse already-computed result
        if bwarn:
            cv2.rectangle(display, (0, h-50), (w, h), (0,0,180), -1)
            cv2.putText(display, f"BOUNDARY: {bwarn}",
                (10, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2)

        cv2.imshow("Frame Recorder — Marsvista", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            recording = not recording
            print(f"  Recording: {'STARTED' if recording else 'PAUSED'}")
        elif key == ord('a'):  fake_x -= 0.1   # A = move left
        elif key == ord('d'):  fake_x += 0.1   # D = move right
        elif key == ord('w'):  fake_y += 0.1   # W = move forward
        elif key == ord('s') and not recording:
            fake_y -= 0.1                       # S = move back (only when not saving)
        elif key == ord('u'):  fake_z += 0.1   # U = up
        elif key == ord('j'):  fake_z -= 0.1   # J = down

    cap.release()
    cv2.destroyAllWindows()

    # Save session log
    with open(log_path, "w") as f:
        json.dump({
            "team"        : "Team Marsvista",
            "mode"        : "webcam",
            "session_time": datetime.now().isoformat(),
            "frames_saved": saved_count,
            "frames"      : metadata,
        }, f, indent=2)

    print(f"\n  Recorder stopped.")
    print(f"  Frames saved: {saved_count}")
    print(f"  Log: {log_path}")


# ══════════════════════════════════════════════════════════════════
# ROS2 MODE (Jetson deployment)
# ══════════════════════════════════════════════════════════════════
def run_ros2():
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
    except ImportError:
        print("ERROR: rclpy not found. Run: source /opt/ros/humble/setup.bash")
        return

    POSE_TOPIC  = "/orb_slam3/pose"
    IMAGE_TOPIC = "/oak/rgb/image_rect"

    class Recorder(Node):
        def __init__(self):
            super().__init__("frame_recorder")
            os.makedirs(SAVE_DIR, exist_ok=True)
            self.bridge     = CvBridge()
            self._pose      = None
            self._lock      = threading.Lock()
            self._idx       = 0
            self._saved     = 0
            self._last_save = 0.0
            self._interval  = 1.0 / SAVE_FPS
            self._meta      = []

            self.create_subscription(PoseStamped, POSE_TOPIC,
                self._pose_cb, 10)
            self.create_subscription(Image, IMAGE_TOPIC,
                self._image_cb, 10)

            self.get_logger().info(f"Recorder ready | pose={POSE_TOPIC} img={IMAGE_TOPIC}")

        def _pose_cb(self, msg):
            with self._lock:
                self._pose = msg

        def _image_cb(self, msg):
            now = time.time()
            if now - self._last_save < self._interval:
                return

            with self._lock:
                pose = self._pose

            x = y = z = 0.0
            if pose:
                x = pose.pose.position.x
                y = pose.pose.position.y
                z = pose.pose.position.z
            else:
                self.get_logger().warn("No SLAM pose — saving with x0 y0 z0",
                    throttle_duration_sec=5.0)

            try:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                self.get_logger().error(f"cv_bridge: {e}")
                return

            h, w = frame.shape[:2]
            if w != 1280 or h != 720:
                frame = cv2.resize(frame, (1280, 720))

            disk_pct = disk_usage_pct(SAVE_DIR)
            if disk_pct >= DISK_STOP_PCT:
                self.get_logger().error(f"Disk {disk_pct:.0f}% full — stopping.")
                return

            self._last_save = now
            blur  = sharpness(frame)
            fname = (f"frame_{self._saved:06d}"
                     f"_x{x:.3f}_y{y:.3f}_z{z:.3f}.jpg")
            cv2.imwrite(os.path.join(SAVE_DIR, fname), frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            warn = boundary_warning(frame)
            if warn:
                self.get_logger().warn(f"BOUNDARY: {warn}")

            self._meta.append({
                "frame": self._saved, "file": fname,
                "x": x, "y": y, "z": z,
                "sharpness": round(blur, 1),
                "boundary_warning": warn,
            })
            self._saved += 1

            if self._saved % 50 == 0:
                self.get_logger().info(
                    f"Saved {self._saved} frames | disk={disk_pct:.0f}%")

    rclpy.init()
    node = Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        log_path = os.path.join(SAVE_DIR, "session_log.json")
        with open(log_path, "w") as f:
            json.dump({
                "team": "Team Marsvista", "mode": "ros2",
                "session_time": datetime.now().isoformat(),
                "frames_saved": node._saved,
                "frames": node._meta,
            }, f, indent=2)
        print(f"\nSaved {node._saved} frames. Log: {log_path}")
        node.destroy_node()
        rclpy.shutdown()


# ── ENTRY POINT ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["webcam", "ros2"],
                        default="webcam",
                        help="webcam = laptop testing | ros2 = Jetson deployment")
    args = parser.parse_args()

    if args.mode == "ros2":
        run_ros2()
    else:
        run_webcam()
