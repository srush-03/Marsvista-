"""
stitch.py — ISRO IRoC-U 2026 ASCEND | Team Marsvista
=====================================================
Builds a full arena mosaic from HD frames saved during flight.

ALGORITHM — SLAM-Guided Incremental Stitching:
    Step 1: Parse SLAM x,y,z pose from every frame filename
    Step 2: Convert world pose to pixel coordinates on mosaic canvas
    Step 3: For each frame compute perspective warp from pose
    Step 4: ORB feature matching between consecutive frames
            refines the pose-based homography (removes SLAM drift)
    Step 5: Alpha blend each frame onto canvas (weighted by overlap count)
    Step 6: Overlay detected feature positions as markers
    Step 7: Save mosaic + coordinate map

WHY NOT cv2.Stitcher:
    cv2.Stitcher fails above ~100 frames due to memory and finds
    feature matches globally (expensive). It has no pose guidance
    so it drifts badly over 3000 frames.

WHY SLAM-GUIDED:
    SLAM gives us approximate world pose per frame.
    We convert pose → canvas pixel transform directly.
    ORB refines only the residual error between adjacent frames.
    Result: fast, stable, handles texture-less regions.

Usage:
    python stitch.py                              # stitch all frames
    python stitch.py --frames flight_frames/      # custom frame dir
    python stitch.py --results results/results.json  # overlay detections
    python stitch.py --every 3                   # use every 3rd frame (faster)
    python stitch.py --scale 0.5                 # 50% output resolution

Output:
    results/mosaic.jpg           ← full arena mosaic
    results/mosaic_annotated.jpg ← mosaic with feature markers
    results/coordinate_map.json  ← pixel ↔ world coordinate mapping
"""

import cv2
import numpy as np
import os
import json
import re
import argparse
import time
from pathlib import Path
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────
FRAMES_DIR      = "flight_frames"
OUT_DIR         = "results"
ARENA_W_M       = 10.7       # arena width in metres (35ft)
ARENA_H_M       = 7.6        # arena height in metres (25ft)
MOSAIC_PPM      = 80         # pixels per metre in output mosaic
BLEND_ALPHA     = 0.6        # weight for new frame over existing mosaic
ORB_MAX_FEAT    = 1000       # ORB features per frame for refinement
ORB_MATCH_RATIO = 0.75       # Lowe's ratio test
MIN_INLIERS     = 8          # minimum RANSAC inliers to accept ORB refinement
EVERY_N         = 2          # use every Nth frame by default


# ── POSE PARSER ───────────────────────────────────────────────────
_RE = re.compile(r"frame_(\d+)_x(-?[\d.]+)_y(-?[\d.]+)_z(-?[\d.]+)")

def parse_pose(fname):
    m = _RE.search(os.path.basename(str(fname)))
    return (int(m[1]), float(m[2]), float(m[3]), float(m[4])) if m else None


# ── POSE → CANVAS TRANSFORM ───────────────────────────────────────
class CanvasMapper:
    """
    Converts SLAM world coordinates (metres) to mosaic canvas pixels.
    Handles coordinate system alignment between SLAM and arena frame.
    """
    def __init__(self, poses, canvas_ppm=MOSAIC_PPM, margin_px=80):
        xs = [p[1] for p in poses]
        ys = [p[2] for p in poses]
        self.min_x  = min(xs) - 0.5
        self.min_y  = min(ys) - 0.5
        self.ppm    = canvas_ppm
        self.margin = margin_px

        # Canvas dimensions
        range_x = (max(xs) - self.min_x + 1.0)
        range_y = (max(ys) - self.min_y + 1.0)
        self.canvas_w = int(range_x * canvas_ppm) + 2 * margin_px
        self.canvas_h = int(range_y * canvas_ppm) + 2 * margin_px

        print(f"  Canvas: {self.canvas_w}x{self.canvas_h}px  "
              f"({range_x:.1f}m x {range_y:.1f}m  @{canvas_ppm}ppm)")

    def world_to_px(self, x, y):
        """World metres → canvas pixel (cx, cy)."""
        cx = int((x - self.min_x) * self.ppm) + self.margin
        cy = int((y - self.min_y) * self.ppm) + self.margin
        return cx, cy

    def px_to_world(self, cx, cy):
        """Canvas pixel → world metres."""
        x = (cx - self.margin) / self.ppm + self.min_x
        y = (cy - self.margin) / self.ppm + self.min_y
        return round(x, 3), round(y, 3)

    def frame_footprint_px(self, x, y, z, frame_hfov_deg=69, frame_vfov_deg=54):
        """
        Returns (w_px, h_px) of the ground footprint of a frame
        captured at altitude z from world position (x,y).
        """
        import math
        gw = 2 * z * math.tan(math.radians(frame_hfov_deg / 2))
        gh = 2 * z * math.tan(math.radians(frame_vfov_deg / 2))
        w_px = int(gw * self.ppm)
        h_px = int(gh * self.ppm)
        return max(w_px, 10), max(h_px, 10)


# ── ORB REFINER ───────────────────────────────────────────────────
class ORBRefiner:
    """
    Refines homography between consecutive frames using ORB features.
    Falls back gracefully if not enough matches found.
    """
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=ORB_MAX_FEAT)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._prev_gray = None
        self._prev_kp   = None
        self._prev_des  = None

    def refine(self, gray_curr, H_init):
        """
        Given initial homography H_init (from pose) and current frame,
        refine using ORB matches with previous frame.
        Returns (H_refined, inlier_count, used_orb)
        """
        kp, des = self.orb.detectAndCompute(gray_curr, None)

        if (des is None or self._prev_des is None or
                len(des) < 10 or len(self._prev_des) < 10):
            self._update(gray_curr, kp, des)
            return H_init, 0, False

        # Match with ratio test
        matches = self.bf.knnMatch(self._prev_des, des, k=2)
        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < ORB_MATCH_RATIO * m[1].distance:
                good.append(m[0])

        if len(good) < MIN_INLIERS:
            self._update(gray_curr, kp, des)
            return H_init, len(good), False

        src = np.float32([self._prev_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])

        H_orb, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0

        self._update(gray_curr, kp, des)

        if H_orb is None or inliers < MIN_INLIERS:
            return H_init, inliers, False

        # Compose: H_init is the pose-based warp, H_orb refines residual
        H_combined = H_init @ np.linalg.inv(H_orb)
        return H_combined, inliers, True

    def _update(self, gray, kp, des):
        self._prev_gray = gray
        self._prev_kp   = kp
        self._prev_des  = des


# ── ALPHA BLEND ───────────────────────────────────────────────────
def blend_onto_canvas(canvas, canvas_mask, warped, warped_mask):
    """
    Alpha-blend warped frame onto canvas.
    Where canvas is empty: just copy.
    Where both overlap: weighted average.
    """
    new_only  = (warped_mask > 0) & (canvas_mask == 0)
    both      = (warped_mask > 0) & (canvas_mask > 0)

    canvas[new_only]  = warped[new_only]
    canvas_mask[new_only] = 255

    if both.any():
        alpha = BLEND_ALPHA
        canvas[both] = (
            canvas[both].astype(np.float32) * (1 - alpha) +
            warped[both].astype(np.float32) * alpha
        ).astype(np.uint8)

    return canvas, canvas_mask


# ── MAIN ──────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.out, exist_ok=True)

    print("\n" + "="*60)
    print("  Arena Mosaic Stitcher — Team Marsvista")
    print("="*60)

    # ── Load frames ───────────────────────────────────────────────
    all_files = sorted(Path(args.frames).glob("*.jpg"))
    valid = [(f, parse_pose(f.name)) for f in all_files if parse_pose(f.name)]

    if not valid:
        print(f"ERROR: No valid frames in {args.frames}")
        return

    # Subsample
    valid = valid[::args.every]
    print(f"  Using {len(valid)} frames (every {args.every} of "
          f"{len(all_files)} total)")

    poses = [p for _, p in valid]

    # ── Build canvas mapper ───────────────────────────────────────
    mapper  = CanvasMapper(poses, canvas_ppm=MOSAIC_PPM)

    # Use median altitude to avoid footprint size variation from noisy SLAM z
    median_z = float(np.median([p[3] for p in poses]))
    print(f"  Median flight altitude: {median_z:.2f}m")
    canvas  = np.zeros((mapper.canvas_h, mapper.canvas_w, 3), dtype=np.uint8)
    c_mask  = np.zeros((mapper.canvas_h, mapper.canvas_w),    dtype=np.uint8)
    refiner = ORBRefiner()

    t0 = time.time()
    orb_used = skip = corrupt = 0

    for fi, (fpath, pose) in enumerate(valid):
        frame_idx, fx, fy, fz = pose

        frame = cv2.imread(str(fpath))
        if frame is None:
            corrupt += 1
            continue

        fh, fw = frame.shape[:2]
        if fw != 1280 or fh != 720:
            frame = cv2.resize(frame, (1280, 720))

        # Resize frame for stitching (save memory)
        if args.scale != 1.0:
            sw = int(fw * args.scale)
            sh = int(fh * args.scale)
            frame = cv2.resize(frame, (sw, sh))
            fw, fh = sw, sh

        # Use median altitude for stable footprint size (avoids SLAM z noise)
        cx, cy   = mapper.world_to_px(fx, fy)
        fw_px, fh_px = mapper.frame_footprint_px(fx, fy, median_z)

        # Build destination corners for homography
        # Frame corners → canvas position based on pose
        dst_corners = np.float32([
            [cx - fw_px//2, cy - fh_px//2],
            [cx + fw_px//2, cy - fh_px//2],
            [cx + fw_px//2, cy + fh_px//2],
            [cx - fw_px//2, cy + fh_px//2],
        ])
        src_corners = np.float32([
            [0, 0], [fw, 0], [fw, fh], [0, fh]
        ])

        H_pose, _ = cv2.findHomography(src_corners, dst_corners)
        if H_pose is None:
            skip += 1
            continue

        # ORB refinement
        gray_curr     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, inliers, used_orb = refiner.refine(gray_curr, H_pose)
        if used_orb:
            orb_used += 1

        # Warp frame onto canvas
        try:
            warped = cv2.warpPerspective(
                frame, H,
                (mapper.canvas_w, mapper.canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
            # Warp a white mask to know where frame lands
            white  = np.ones((fh, fw), dtype=np.uint8) * 255
            w_mask = cv2.warpPerspective(
                white, H,
                (mapper.canvas_w, mapper.canvas_h),
                flags=cv2.INTER_NEAREST,
            )
        except cv2.error as e:
            skip += 1
            continue

        canvas, c_mask = blend_onto_canvas(canvas, c_mask, warped, w_mask)

        if fi % 200 == 0:
            elapsed = time.time() - t0
            pct = 100 * fi / len(valid)
            coverage = 100 * (c_mask > 0).sum() / (mapper.canvas_w * mapper.canvas_h)
            print(f"  [{fi:5d}/{len(valid)}] {pct:.0f}%  "
                  f"coverage={coverage:.0f}%  orb={orb_used}  t={elapsed:.0f}s")

    elapsed = time.time() - t0
    coverage = 100*(c_mask>0).sum()/(mapper.canvas_w*mapper.canvas_h)
    print(f"\n  Stitching complete in {elapsed:.1f}s")
    print(f"  Coverage: {coverage:.1f}%")
    print(f"  ORB refinements: {orb_used}/{len(valid)}")
    print(f"  Skipped: {skip}  Corrupt: {corrupt}")

    # ── Fill gaps with inpainting ─────────────────────────────────
    if (c_mask == 0).any():
        gap_mask = (c_mask == 0).astype(np.uint8) * 255
        canvas   = cv2.inpaint(canvas, gap_mask, 3, cv2.INPAINT_NS)
        print(f"  Gap inpainting applied.")

    # ── Save plain mosaic ─────────────────────────────────────────
    mosaic_path = os.path.join(args.out, "mosaic.jpg")
    cv2.imwrite(mosaic_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Mosaic saved: {mosaic_path}")

    # ── Annotate with detections ──────────────────────────────────
    annotated = canvas.copy()
    coord_map  = {
        "ppm"    : mapper.ppm,
        "min_x"  : mapper.min_x,
        "min_y"  : mapper.min_y,
        "margin" : mapper.margin,
        "canvas_w": mapper.canvas_w,
        "canvas_h": mapper.canvas_h,
    }

    if args.results and os.path.exists(args.results):
        with open(args.results) as f:
            res = json.load(f)

        colours = {
            0: (0,   255,   0),    # green
            1: (0,   200, 255),    # cyan
            2: (255, 128,   0),    # orange
            3: (255,   0, 255),    # magenta
            4: (255, 255,   0),    # yellow
        }

        instances = res.get("instances", [])
        print(f"\n  Annotating {len(instances)} detected instances...")

        for ii, inst in enumerate(instances):
            coords = inst["coordinates"]
            cx, cy = mapper.world_to_px(coords["x"], coords["y"])
            col    = colours[ii % len(colours)]
            label  = f"{inst['feature_type']} #{inst['instance_index']}"
            score  = inst["confidence"]["cnn_score"]

            # Draw crosshair
            r = 25
            cv2.circle(annotated, (cx,cy), r, col, 2)
            cv2.line(annotated, (cx-r-5,cy),(cx+r+5,cy), col, 2)
            cv2.line(annotated, (cx,cy-r-5),(cx,cy+r+5), col, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label,
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated,
                (cx+r+2, cy-th-4),(cx+r+tw+6, cy+4), (0,0,0), -1)
            cv2.putText(annotated, label,
                (cx+r+4, cy), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, col, 2)
            cv2.putText(annotated, f"CNN={score:.2f}",
                (cx+r+4, cy+18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, col, 1)

        # Legend
        cv2.rectangle(annotated,(5,5),(280,30+25*len(instances)),(0,0,0),-1)
        cv2.putText(annotated,"Team Marsvista — ASCEND",(10,22),
            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        for ii,inst in enumerate(instances):
            col = colours[ii % len(colours)]
            cv2.putText(annotated,
                f"{inst['instance_id']}  ({inst['coordinates']['x']}m, {inst['coordinates']['y']}m)",
                (10,47+25*ii),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)

    ann_path = os.path.join(args.out, "mosaic_annotated.jpg")
    cv2.imwrite(ann_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Annotated mosaic: {ann_path}")

    # ── Coordinate map ────────────────────────────────────────────
    cmap_path = os.path.join(args.out, "coordinate_map.json")
    with open(cmap_path, "w") as f:
        json.dump({
            "team"        : "Team Marsvista",
            "created_at"  : datetime.now().isoformat(),
            "mosaic_file" : "mosaic.jpg",
            "canvas"      : coord_map,
            "usage"       : {
                "world_to_px" : "cx = int((x - min_x) * ppm) + margin",
                "px_to_world" : "x = (cx - margin) / ppm + min_x",
            },
        }, f, indent=2)
    print(f"  Coordinate map: {cmap_path}")

    print(f"\n{'='*60}")
    print(f"  Mosaic: {mosaic_path}")
    print(f"  Annotated: {ann_path}")
    print(f"  Coverage: {coverage:.1f}%")
    print("="*60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames",  default=FRAMES_DIR)
    p.add_argument("--out",     default=OUT_DIR)
    p.add_argument("--results", default="results/results.json",
                   help="Path to results.json for detection overlay")
    p.add_argument("--every",   type=int, default=EVERY_N,
                   help="Use every Nth frame (default 2)")
    p.add_argument("--scale",   type=float, default=0.5,
                   help="Scale frames before stitching (default 0.5)")
    main(p.parse_args())
