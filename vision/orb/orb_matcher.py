import cv2
import numpy as np


class ORBMatcher:

    def __init__(self, seed_loader, threshold=25):

        self.seed_desc = seed_loader.get_descriptors()

        self.orb = cv2.ORB_create(1000)

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True
        )

        self.threshold = threshold

        print("[ORB] Matcher ready")


    def match_frame(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp, desc = self.orb.detectAndCompute(gray, None)

        detections = []

        if desc is None:
            return detections

        for feature_type, seed_desc in self.seed_desc.items():

            matches = self.matcher.match(desc, seed_desc)

            score = len(matches)

            if score > self.threshold:

                detections.append({

                    "feature_type": feature_type,
                    "confidence": score,
                    "bbox": (0,0,frame.shape[1],frame.shape[0]),
                    "bbox_center": (
                        frame.shape[1]//2,
                        frame.shape[0]//2
                    )
                })

        return detections