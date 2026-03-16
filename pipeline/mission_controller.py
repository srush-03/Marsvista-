import cv2
import time
import argparse
import threading
import queue
import os

from config.config_loader import ConfigLoader
from vision.detector_factory import DetectorFactory
from vision.yellow_line_detector import YellowLineDetector
from vision.instance_tracker import InstanceTracker
from mission_logging.metadata_logger import MetadataLogger


class MissionController:

    def __init__(self, seeds_folder=None):

        config_loader = ConfigLoader()

        self.config = config_loader.get_all()

        if seeds_folder:
            self.config["seed_path"] = seeds_folder

        print("\n=================================")
        print(" AUTONOMOUS DRONE MISSION START ")
        print("=================================\n")

        os.makedirs("data/detections", exist_ok=True)
        os.makedirs("data/keyframes", exist_ok=True)
        os.makedirs("data/logs", exist_ok=True)

        self.detector = DetectorFactory.create_detector()
        self.yellow_detector = YellowLineDetector()
        self.tracker = InstanceTracker()
        self.logger = MetadataLogger("data/logs")

        width = self.config.get("frame_width", 1280)
        height = self.config.get("frame_height", 720)
        fps = self.config.get("camera_fps", 30)

        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError("Camera failed to start")

        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)

        self.running = True

        self.frame_index = 0

        self.camera_thread = threading.Thread(
            target=self.camera_loop, daemon=True)

        self.detector_thread = threading.Thread(
            target=self.detector_loop, daemon=True)


    def camera_loop(self):

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                continue

            if not self.frame_queue.full():
                self.frame_queue.put(frame)


    def detector_loop(self):

        while self.running:

            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            detections = self.detector.match_frame(frame)

            if not self.result_queue.full():
                self.result_queue.put((frame, detections))


    def run(self):

        self.camera_thread.start()
        self.detector_thread.start()

        prev_time = time.time()

        while True:

            try:
                frame, detections = self.result_queue.get(timeout=1)
            except queue.Empty:
                continue

            self.frame_index += 1

            display = frame.copy()

            for det in detections:

                ftype = det["feature_type"]
                confidence = det["confidence"]

                x, y, w, h = det["bbox"]
                center = det["bbox_center"]

                instance_id, is_new = self.tracker.update(
                    ftype,
                    center,
                    confidence,
                    self.frame_index
                )

                det_dir = f"data/detections/{ftype}"
                os.makedirs(det_dir, exist_ok=True)

                det_path = f"{det_dir}/det_{instance_id}_{self.frame_index}.jpg"

                cv2.imwrite(det_path, frame)

                self.logger.log_detection(
                    frame_index=self.frame_index,
                    feature_type=ftype,
                    detection=det,
                    image_path=det_path,
                    drone_pose={"x":0,"y":0,"z":4.5},
                    instance_id=instance_id,
                    is_new_instance=is_new
                )

                color = (0,255,0) if is_new else (0,200,255)

                cv2.rectangle(display,(x,y),(x+w,y+h),color,2)

                cv2.putText(display,
                    f"{instance_id} {confidence:.2f}",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2)

            boundary = self.yellow_detector.detect(frame)

            if boundary["detected"]:

                for line in boundary["lines"]:
                    x1,y1,x2,y2 = line[0]
                    cv2.line(display,(x1,y1),(x2,y2),(0,255,255),3)

                cv2.putText(display,
                    f"BOUNDARY: {boundary['direction_warning']}",
                    (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,255),
                    2)

            now = time.time()
            fps = 1/(now-prev_time+1e-6)
            prev_time = now

            cv2.putText(display,
                f"Frame: {self.frame_index}",
                (10,display.shape[0]-60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200,200,200),
                1)

            cv2.putText(display,
                f"FPS: {fps:.1f}",
                (10,display.shape[0]-30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                1)

            cv2.imshow("Drone Perception System",display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.shutdown()


    def shutdown(self):

        print("\nShutting down mission...")

        self.running = False

        if hasattr(self.detector,"stop"):
            self.detector.stop()

        self.cap.release()

        cv2.destroyAllWindows()

        summary = self.tracker.get_summary()

        self.logger.log_instance_summary(summary)

        print("Mission data saved.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=str,
        default=None
    )

    args = parser.parse_args()

    controller = MissionController(
        seeds_folder=args.seeds
    )

    controller.run()