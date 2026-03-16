import cv2
import numpy as np


class YellowLineDetector:

    def detect(self, frame):

        height, width = frame.shape[:2]

        result = {
            "detected": False,
            "lines": None,
            "position": None,
            "direction_warning": None
        }

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([18, 80, 80])
        upper = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((3,3), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        ground_mask = np.zeros_like(mask)
        ground_mask[height//2:, :] = mask[height//2:, :]

        edges = cv2.Canny(ground_mask, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi/180,
            threshold=60,
            minLineLength=80,
            maxLineGap=30
        )

        if lines is None:
            return result

        long_lines = []

        for line in lines:

            x1, y1, x2, y2 = line[0]

            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            if length > width * 0.15:
                long_lines.append(line)

        if not long_lines:
            return result

        all_x = []
        all_y = []

        for line in long_lines:

            x1,y1,x2,y2 = line[0]

            all_x.extend([x1,x2])
            all_y.extend([y1,y2])

        avg_x = np.mean(all_x)
        avg_y = np.mean(all_y)

        left_zone = width * 0.3
        right_zone = width * 0.7
        bottom_zone = height * 0.75

        if avg_x < left_zone:

            position = "left"
            warning = "TURN_RIGHT"

        elif avg_x > right_zone:

            position = "right"
            warning = "TURN_LEFT"

        elif avg_y > bottom_zone:

            position = "bottom"
            warning = "TURN_BACK"

        else:

            position = "ahead"
            warning = "CAUTION"

        result["detected"] = True
        result["lines"] = long_lines
        result["position"] = position
        result["direction_warning"] = warning

        return result