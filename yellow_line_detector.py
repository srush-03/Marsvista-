import cv2
import numpy as np

class YellowLineDetector:
    def detect(self, frame):
        height, width = frame.shape[:2]

        result = {
            'detected': False,
            'confidence': 0.0,
            'direction_warning': None,
            'lines': None,
            'position': None  # 'left', 'right', 'top', 'bottom'
        }

        # ── STEP 1: ISOLATE YELLOW COLOR ──────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Yellow range — covers different lighting conditions
        lower_yellow = np.array([18, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # ── STEP 2: REMOVE NOISE ───────────────────────
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # ── STEP 3: LOOK ONLY IN BOTTOM HALF OF FRAME ─
        # The ground line will ALWAYS appear in the bottom
        # portion of the frame when drone looks downward.
        # This single rule eliminates 90% of false positives
        # because yellow objects in the air or at drone height
        # appear in the upper half.
        ground_mask = np.zeros_like(mask)
        ground_mask[height//2:, :] = mask[height//2:, :]

        # ── STEP 4: FIND CONTOURS AND FILTER BY SHAPE ─
        contours, _ = cv2.findContours(
            ground_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        valid_line_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Ignore tiny specks
            if area < 300:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)

            # ── KEY SHAPE FILTER ──────────────────────
            # A boundary LINE has aspect ratio >> 1
            # meaning much wider than tall (or much taller than wide)
            # A yellow OBJECT or PATCH has aspect ratio close to 1
            # (roughly square or round)

            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

            # Only accept if very elongated — ratio > 4 means
            # it is at least 4x longer than it is wide
            # This rejects square patches, round objects, equipment
            if aspect_ratio < 4.0:
                continue

            # Also check that the contour is thin relative to length
            # Filled area vs bounding box area
            # A line fills maybe 20-40% of its bounding box
            # A solid patch fills 70-100% of its bounding box
            bbox_area = w * h
            fill_ratio = area / (bbox_area + 1e-5)

            # Reject solid filled shapes (patches, tiles, equipment)
            if fill_ratio > 0.6:
                continue

            valid_line_contours.append((cnt, x, y, w, h, aspect_ratio))

        if not valid_line_contours:
            return result

        # ── STEP 5: DETECT LINES USING HOUGH ──────────
        edges = cv2.Canny(ground_mask, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=60,
            minLineLength=80,   # must be long — rejects short objects
            maxLineGap=30
        )

        if lines is None or len(lines) < 1:
            return result

        # ── STEP 6: FILTER LINES BY LENGTH ────────────
        # Only keep lines that are genuinely long
        # Short lines = noise or object edges
        # Long lines = boundary
        long_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > width * 0.15:  # at least 15% of frame width
                long_lines.append(line)

        if not long_lines:
            return result

        # ── STEP 7: DETERMINE POSITION AND DIRECTION ──
        # Where in the frame is the line?
        # This tells the drone which way to turn

        all_x = []
        all_y = []
        for line in long_lines:
            x1, y1, x2, y2 = line[0]
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

        avg_x = np.mean(all_x)
        avg_y = np.mean(all_y)

        # Divide frame into zones
        left_zone   = width * 0.3
        right_zone  = width * 0.7
        bottom_zone = height * 0.75

        # Determine which boundary edge is visible
        if avg_x < left_zone:
            position  = 'left'
            warning   = 'TURN_RIGHT'   # line on left → turn right
        elif avg_x > right_zone:
            position  = 'right'
            warning   = 'TURN_LEFT'    # line on right → turn left
        elif avg_y > bottom_zone:
            position  = 'bottom'
            warning   = 'TURN_BACK'    # line at bottom → turn back
        else:
            position  = 'ahead'
            warning   = 'CAUTION'      # line somewhere ahead

        confidence = min(len(long_lines) / 3.0, 1.0)

        result['detected']          = True
        result['confidence']        = round(confidence, 2)
        result['direction_warning'] = warning
        result['position']          = position
        result['lines']             = long_lines

        return result


# ── TEST ──────────────────────────────────────────
if __name__ == "__main__":
    import sys

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Yellow Line Detector Test")
    print("Place a long yellow strip/tape on the floor")
    print("Move camera toward it from above")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display   = frame.copy()
        height, width = frame.shape[:2]

        # Show the bottom half zone boundary
        cv2.line(display,
            (0, height//2), (width, height//2),
            (255, 0, 0), 1)
        cv2.putText(display,
            "Ground zone (detection area)",
            (10, height//2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        result = YellowLineDetector().detect(frame)

        if result['detected']:
            # Draw detected lines
            for line in result['lines']:
                x1, y1, x2, y2 = line[0]
                cv2.line(display, (x1,y1), (x2,y2), (0,255,255), 4)

            cv2.putText(display,
                f"BOUNDARY DETECTED | {result['direction_warning']} "
                f"| conf: {result['confidence']}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            print(f"LINE DETECTED | position: {result['position']} "
                  f"| warning: {result['direction_warning']} "
                  f"| confidence: {result['confidence']}")
        else:
            cv2.putText(display,
                "No boundary line detected",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Yellow Line Detector Test", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()