import numpy as np
import threading
import time


class CNNMatcher:

    def __init__(self, cnn_loader, threshold=0.65):

        self.seed_vectors = cnn_loader.get_vectors()

        self.loader = cnn_loader

        self.threshold = threshold

        self._latest_frame = None
        self._latest_dets = []

        self._running = True

        self._lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True
        )

        self._thread.start()

        print("[CNN] Matcher ready")


    def submit_frame(self, frame):

        with self._lock:
            self._latest_frame = frame.copy()


    def match_frame(self):

        with self._lock:
            return list(self._latest_dets)


    def stop(self):

        self._running = False


    def _loop(self):

        while self._running:

            with self._lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.01)
                continue

            detections = self._process_frame(frame)

            with self._lock:
                self._latest_dets = detections


    def _process_frame(self, frame):

        vec = self.loader.extract_vector(frame)

        detections = []

        for feature_type, seed_vec in self.seed_vectors.items():

            similarity = float(np.dot(vec, seed_vec))

            if similarity >= self.threshold:

                detections.append({

                    "feature_type": feature_type,
                    "confidence": similarity,
                    "bbox": (0,0,frame.shape[1],frame.shape[0]),
                    "bbox_center": (
                        frame.shape[1]//2,
                        frame.shape[0]//2
                    )
                })

        return detections