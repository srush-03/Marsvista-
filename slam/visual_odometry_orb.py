import cv2
import numpy as np


class VisualOdometryORB:

    def __init__(self):

        self.orb = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_kp = None
        self.prev_desc = None

        self.x = 0
        self.y = 0

        print("[SLAM] ORB Visual Odometry ready")


    def process_frame(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp, desc = self.orb.detectAndCompute(gray, None)

        if self.prev_desc is None:

            self.prev_kp = kp
            self.prev_desc = desc
            return self.get_pose()

        matches = self.matcher.match(desc, self.prev_desc)

        matches = sorted(matches, key=lambda x: x.distance)

        good = matches[:40]

        if len(good) < 8:
            return self.get_pose()

        pts1 = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pts2 = np.float32([self.prev_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        matrix, _ = cv2.estimateAffinePartial2D(pts2, pts1)

        if matrix is not None:

            dx = matrix[0,2]
            dy = matrix[1,2]

            self.x += dx
            self.y += dy

        self.prev_kp = kp
        self.prev_desc = desc

        return self.get_pose()


    def get_pose(self):

        return {
            "x": self.x,
            "y": self.y
        }