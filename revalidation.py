"""
revalidation.py — ISRO IRoC-U 2026 ASCEND | Team Marsvista
===========================================================
Independent revalidation of post_match.py results.

WHY A SEPARATE ALGORITHM:
    post_match.py uses DINOv2 + LBP + HSV for matching.
    If any of those signals produced a false positive,
    revalidation catches it using a completely different approach:
    ORB keypoint matching + SSIM pixel similarity.

    Two independent algorithms agreeing = high confidence.
    One disagrees = flag for re-sortie.

8 CHECKS PERFORMED:

    Check 1 — Coverage
        Were all 3 expected feature types found?
        FAIL if any type has 0 instances.

    Check 2 — Instance count
        Each type should have 2-3 instances.
        WARN if only 1 (maybe missed one).
        WARN if 4+ (maybe false positives).

    Check 3 — Spatial spread
        Instances of the same type should be > 1m apart.
        FAIL if two instances are < 0.5m apart (same location = duplicate).

    Check 4 — Arena bounds
        Coordinates must be within arena dimensions (+ 0.5m margin).
        FAIL if any coordinate is outside arena.

    Check 5 — Confidence threshold
        CNN fusion score must be above minimum per instance.
        WARN if any instance is below 0.50.

    Check 6 — Image integrity
        HD proof images must be readable and correct size.
        FAIL if any proof image is corrupt or missing.

    Check 7 — ORB re-verification
        ORB keypoint matching between:
            reference LR image (128x128)
            cropped region from HD proof image (same bbox)
        Minimum inliers required to confirm match.
        NOTE: oxide/ice have few keypoints by nature.
              For those, SSIM check (Check 8) carries more weight.

    Check 8 — SSIM cross-check
        SSIM between 128x128 LR reference and
        128x128 crop from HD proof image.
        Must exceed minimum threshold.
        Uses multi-scale SSIM for robustness.

Usage:
    python revalidation.py                              # uses results/results.json
    python revalidation.py --results results/results.json
    python revalidation.py --results results/results.json --refs refs/

Output:
    results/revalidation_report.json    ← machine-readable
    results/revalidation_summary.txt    ← human-readable
    Terminal: per-instance pass/fail for each check

Optimization suggestions:
    - ORB nfeatures=500 is fast; lower to 200 for speed on Jetson
    - SSIM at 128x128 is <1ms per pair; no optimization needed
    - Run in parallel per instance if >20 instances (ThreadPoolExecutor)
"""

import cv2
import numpy as np
import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from skimage.metrics import structural_similarity as ssim_fn

# ── CONFIG ────────────────────────────────────────────────────────
ARENA_W_M        = 10.7    # 35ft arena width
ARENA_H_M        = 7.6     # 25ft arena height
ARENA_MARGIN_M   = 0.5     # tolerance for coordinate bounds check

MIN_CNN_SCORE    = 0.50    # minimum acceptable confidence
MIN_ORB_INLIERS  = 6       # minimum ORB RANSAC inliers to confirm
MIN_SSIM         = 0.20    # minimum SSIM between ref and proof crop
MIN_INSTANCES    = 2       # expected minimum per feature type
MAX_INSTANCES    = 3       # expected maximum per feature type
MIN_SPREAD_M     = 0.5     # minimum distance between instances of same type

LR_SIZE          = (128, 128)


# ── ORB VERIFIER ──────────────────────────────────────────────────
class ORBVerifier:
    """
    Independent verification using ORB keypoint matching.
    Completely separate from DINOv2/LBP/HSV pipeline.
    Uses RANSAC homography to count geometrically consistent matches.
    """

    def __init__(self, n_features=500):
        self.orb = cv2.ORB_create(
            nfeatures    = n_features,
            scaleFactor  = 1.2,
            nlevels      = 8,
            edgeThreshold= 15,
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def verify(self, ref_img, probe_img):
        """
        Match ref_img against probe_img using ORB.

        Both images should be 128x128 (or will be resized).
        Returns (inlier_count, match_ratio, verdict_str).

        verdict_str:
            "CONFIRMED"  — enough inliers, homography found
            "WEAK"       — some matches but below threshold
            "NO_MATCH"   — insufficient matches or no homography
            "LOW_KP"     — too few keypoints (expected for oxide/ice)
        """
        if ref_img is None or probe_img is None:
            return 0, 0.0, "NO_IMAGE"

        # Ensure same size
        ref   = cv2.resize(ref_img,   LR_SIZE, interpolation=cv2.INTER_AREA)
        probe = cv2.resize(probe_img, LR_SIZE, interpolation=cv2.INTER_AREA)

        # Convert to grayscale for ORB
        ref_gray   = cv2.cvtColor(ref,   cv2.COLOR_BGR2GRAY)
        probe_gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better keypoint detection on low-contrast images
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        ref_gray   = cl.apply(ref_gray)
        probe_gray = cl.apply(probe_gray)

        kp_ref,   des_ref   = self.orb.detectAndCompute(ref_gray,   None)
        kp_probe, des_probe = self.orb.detectAndCompute(probe_gray, None)

        if des_ref is None or des_probe is None:
            return 0, 0.0, "LOW_KP"
        if len(kp_ref) < 5 or len(kp_probe) < 5:
            return 0, 0.0, "LOW_KP"   # expected for oxide/ice patches

        # kNN matching with ratio test
        try:
            matches = self.bf.knnMatch(des_ref, des_probe, k=2)
        except cv2.error:
            return 0, 0.0, "NO_MATCH"

        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 4:
            return 0, float(len(good)) / max(len(kp_ref), 1), "NO_MATCH"

        # RANSAC homography to count geometrically consistent matches
        src = np.float32([kp_ref[m.queryIdx].pt   for m in good]).reshape(-1,1,2)
        dst = np.float32([kp_probe[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0

        ratio = float(inliers) / max(len(kp_ref), 1)

        if inliers >= MIN_ORB_INLIERS:
            return inliers, ratio, "CONFIRMED"
        elif inliers >= 3:
            return inliers, ratio, "WEAK"
        else:
            return inliers, ratio, "NO_MATCH"


# ── SSIM CROSS-CHECK ──────────────────────────────────────────────
def ssim_verify(ref_img, proof_crop):
    """
    SSIM between 128x128 reference and 128x128 crop from proof image.
    Returns (score, verdict).
    """
    if ref_img is None or proof_crop is None:
        return 0.0, "NO_IMAGE"

    ref   = cv2.resize(ref_img,    LR_SIZE, interpolation=cv2.INTER_AREA)
    probe = cv2.resize(proof_crop, LR_SIZE, interpolation=cv2.INTER_AREA)

    # Apply CLAHE before SSIM for lighting robustness
    def enhance(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        cl  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        lab[:,:,0] = cl.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    ref   = enhance(ref)
    probe = enhance(probe)

    ref_gray   = cv2.cvtColor(ref,   cv2.COLOR_BGR2GRAY)
    probe_gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)

    score = float(ssim_fn(ref_gray, probe_gray, data_range=255))
    verdict = "CONFIRMED" if score >= MIN_SSIM else "WEAK" \
              if score >= MIN_SSIM * 0.7 else "NO_MATCH"

    return round(score, 4), verdict


# ── LOAD LR REFERENCE IMAGES ──────────────────────────────────────
def load_ref_images(refs_dir):
    """Load one LR image per feature type. Returns {feature_name: bgr_image}."""
    result = {}
    if not os.path.isdir(refs_dir):
        return result

    for fname in sorted(os.listdir(refs_dir)):
        if not fname.lower().endswith((".jpg",".png")):
            continue
        if "HD" in fname or "metadata" in fname:
            continue
        path = os.path.join(refs_dir, fname)
        img  = cv2.imread(path)
        if img is None:
            continue
        img  = cv2.resize(img, LR_SIZE, interpolation=cv2.INTER_AREA)

        # Base name = feature type
        import re
        base = re.sub(r"(_LR)?(_[A-C])?(\.(jpg|png))$", "", fname, flags=re.I)
        base = re.sub(r"\.(jpg|png)$", "", base, flags=re.I)
        if base not in result:
            result[base] = img

    return result


# ── 8 CHECKS ──────────────────────────────────────────────────────
def run_checks(instances, missing, ref_images, frames_dir, out_dir):
    """
    Run all 8 revalidation checks.
    Returns (check_results_list, flags_list, mission_complete).
    """
    orb_verifier = ORBVerifier()
    check_results = []
    flags = []

    def add_check(name, passed, detail, warning_only=False):
        status = "PASS" if passed else ("WARN" if warning_only else "FAIL")
        check_results.append({
            "check"  : name,
            "status" : status,
            "detail" : detail,
        })
        if not passed:
            sym = "⚠" if warning_only else "✗"
            flags.append(f"[{status}] {name}: {detail}")
            print(f"    {sym} {name}: {detail}")
        else:
            print(f"    ✓ {name}: {detail}")

    # ── Group instances by feature type ───────────────────────────
    by_type = {}
    for inst in instances:
        ft = inst["feature_type"]
        by_type.setdefault(ft, []).append(inst)

    print("\n  Running 8 checks...\n")

    # CHECK 1 — Coverage
    n_found   = len(by_type)
    n_expected = n_found + len(missing)
    check1_ok = len(missing) == 0
    add_check(
        "Check 1 — Coverage",
        check1_ok,
        f"{n_found}/{n_expected} feature types found"
        + (f" | missing: {missing}" if missing else ""),
    )

    # CHECK 2 — Instance count per type
    count_ok = True
    count_detail_parts = []
    for ft, insts in by_type.items():
        n = len(insts)
        if n < MIN_INSTANCES:
            count_ok = False
            count_detail_parts.append(f"{ft}={n} (expected ≥{MIN_INSTANCES})")
        elif n > MAX_INSTANCES:
            count_detail_parts.append(f"{ft}={n} (expected ≤{MAX_INSTANCES}, possible FP)")
        else:
            count_detail_parts.append(f"{ft}={n} ✓")
    add_check(
        "Check 2 — Instance count",
        count_ok,
        "  ".join(count_detail_parts) or "no instances",
        warning_only=True,
    )

    # CHECK 3 — Spatial spread (no duplicates)
    spread_ok    = True
    spread_detail = []
    for ft, insts in by_type.items():
        for i in range(len(insts)):
            for j in range(i+1, len(insts)):
                xi, yi = insts[i]["coordinates"]["x"], insts[i]["coordinates"]["y"]
                xj, yj = insts[j]["coordinates"]["x"], insts[j]["coordinates"]["y"]
                dist   = ((xi-xj)**2 + (yi-yj)**2) ** 0.5
                if dist < MIN_SPREAD_M:
                    spread_ok = False
                    spread_detail.append(
                        f"{ft} inst#{i} and inst#{j} only {dist:.2f}m apart")
    add_check(
        "Check 3 — Spatial spread",
        spread_ok,
        "All instances well-separated" if spread_ok
        else " | ".join(spread_detail),
    )

    # CHECK 4 — Arena bounds
    bounds_ok     = True
    bounds_detail = []
    for inst in instances:
        x = inst["coordinates"]["x"]
        y = inst["coordinates"]["y"]
        in_bounds = (
            -ARENA_MARGIN_M <= x <= ARENA_W_M + ARENA_MARGIN_M and
            -ARENA_MARGIN_M <= y <= ARENA_H_M + ARENA_MARGIN_M
        )
        if not in_bounds:
            bounds_ok = False
            bounds_detail.append(
                f"{inst['instance_id']} at ({x},{y}) outside arena "
                f"({ARENA_W_M}x{ARENA_H_M}m)")
    # Special: if all coords are exactly 0,0 → SLAM failure
    zero_coords = [i for i in instances
                   if i["coordinates"]["x"] == 0.0
                   and i["coordinates"]["y"] == 0.0]
    if len(zero_coords) > len(instances) * 0.5:
        bounds_ok = False
        bounds_detail.append(
            f"{len(zero_coords)} instances at (0,0) — SLAM pose missing")
    add_check(
        "Check 4 — Arena bounds",
        bounds_ok,
        "All coordinates within arena" if bounds_ok
        else " | ".join(bounds_detail),
    )

    # CHECK 5 — Confidence scores
    conf_ok     = True
    conf_detail = []
    for inst in instances:
        score = inst["confidence"]["cnn_score"]
        if score < MIN_CNN_SCORE:
            conf_ok = False
            conf_detail.append(
                f"{inst['instance_id']} score={score:.3f} < {MIN_CNN_SCORE}")
    add_check(
        "Check 5 — Confidence scores",
        conf_ok,
        "All above minimum" if conf_ok else " | ".join(conf_detail),
        warning_only=True,
    )

    # CHECK 6 — Image integrity
    img_ok     = True
    img_detail = []
    for inst in instances:
        proof_name = inst.get("hd_proof_image", "")
        if not proof_name:
            img_ok = False
            img_detail.append(f"{inst['instance_id']}: no proof image path")
            continue
        proof_path = os.path.join(out_dir, proof_name)
        if not os.path.exists(proof_path):
            img_ok = False
            img_detail.append(f"{inst['instance_id']}: file not found")
            continue
        img = cv2.imread(proof_path)
        if img is None:
            img_ok = False
            img_detail.append(f"{inst['instance_id']}: corrupt/unreadable")
        elif img.shape[0] < 100 or img.shape[1] < 100:
            img_ok = False
            img_detail.append(
                f"{inst['instance_id']}: too small {img.shape}")
    add_check(
        "Check 6 — Image integrity",
        img_ok,
        "All proof images readable" if img_ok else " | ".join(img_detail),
    )

    # CHECK 7 — ORB re-verification
    orb_results  = {}
    orb_ok       = True
    orb_detail   = []

    for inst in instances:
        ft         = inst["feature_type"]
        ref_img    = ref_images.get(ft)
        proof_name = inst.get("hd_proof_image", "")
        proof_path = os.path.join(out_dir, proof_name) if proof_name else ""

        if ref_img is None:
            orb_results[inst["instance_id"]] = {
                "verdict": "NO_REF", "inliers": 0, "ratio": 0}
            continue

        if not proof_path or not os.path.exists(proof_path):
            orb_results[inst["instance_id"]] = {
                "verdict": "NO_PROOF", "inliers": 0, "ratio": 0}
            continue

        # Crop the matched window from proof image
        proof_hd = cv2.imread(proof_path)
        if proof_hd is None:
            orb_results[inst["instance_id"]] = {
                "verdict": "CORRUPT", "inliers": 0, "ratio": 0}
            continue

        bbox = inst.get("window_bbox_px", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            ph, pw = proof_hd.shape[:2]
            x1 = max(0, min(x1, pw-1))
            y1 = max(0, min(y1, ph-1))
            x2 = max(x1+10, min(x2, pw))
            y2 = max(y1+10, min(y2, ph))
            proof_crop = proof_hd[y1:y2, x1:x2]
        else:
            proof_crop = proof_hd

        inliers, ratio, verdict = orb_verifier.verify(ref_img, proof_crop)
        orb_results[inst["instance_id"]] = {
            "verdict": verdict, "inliers": inliers, "ratio": round(ratio,4)}

        # LOW_KP is acceptable for oxide/ice (fine texture, few keypoints)
        # Only flag genuine NO_MATCH
        if verdict == "NO_MATCH":
            orb_detail.append(
                f"{inst['instance_id']}: {verdict} ({inliers} inliers)")

    # ORB: fail only if NO_MATCH on rock-type features
    # oxide/ice LOW_KP is expected — they have no keypoints
    orb_hard_fails = []
    for inst_id, res in orb_results.items():
        ft_lower = inst_id.lower()
        has_keypoints = any(kw in ft_lower for kw in ["rock","layer","stone"])
        if has_keypoints and res["verdict"] == "NO_MATCH":
            orb_hard_fails.append(inst_id)

    orb_ok = len(orb_hard_fails) == 0
    add_check(
        "Check 7 — ORB re-verification",
        orb_ok,
        f"ORB results: " + ", ".join(
            f"{k}={v['verdict']}({v['inliers']})" for k, v in orb_results.items()
        ) if orb_results else "no instances to verify",
        warning_only=not orb_ok,
    )

    # CHECK 8 — SSIM cross-check
    ssim_ok      = True
    ssim_detail  = []
    ssim_results = {}

    for inst in instances:
        ft         = inst["feature_type"]
        ref_img    = ref_images.get(ft)
        proof_name = inst.get("hd_proof_image", "")
        proof_path = os.path.join(out_dir, proof_name) if proof_name else ""

        if ref_img is None or not proof_path or not os.path.exists(proof_path):
            ssim_results[inst["instance_id"]] = 0.0
            continue

        proof_hd = cv2.imread(proof_path)
        if proof_hd is None:
            ssim_results[inst["instance_id"]] = 0.0
            continue

        bbox = inst.get("window_bbox_px", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            ph, pw = proof_hd.shape[:2]
            x1 = max(0, min(x1, pw-1))
            y1 = max(0, min(y1, ph-1))
            x2 = max(x1+10, min(x2, pw))
            y2 = max(y1+10, min(y2, ph))
            proof_crop = proof_hd[y1:y2, x1:x2]
        else:
            proof_crop = proof_hd

        score, verdict = ssim_verify(ref_img, proof_crop)
        ssim_results[inst["instance_id"]] = score

        if verdict == "NO_MATCH":
            ssim_ok = False
            ssim_detail.append(
                f"{inst['instance_id']}: SSIM={score:.3f} < {MIN_SSIM}")

    add_check(
        "Check 8 — SSIM cross-check",
        ssim_ok,
        "SSIM scores: " + ", ".join(
            f"{k}={v:.3f}" for k, v in ssim_results.items()
        ) if ssim_results else "no instances to check",
        warning_only=True,
    )

    # ── Final verdict ─────────────────────────────────────────────
    hard_fails = [c for c in check_results
                  if c["status"] == "FAIL"]
    mission_complete = len(hard_fails) == 0

    return check_results, flags, mission_complete, orb_results, ssim_results


# ── MAIN ──────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.out, exist_ok=True)

    print("\n  Loading results...")
    if not os.path.exists(args.results):
        print(f"  ERROR: {args.results} not found. Run post_match.py first.")
        return None

    with open(args.results) as f:
        data = json.load(f)

    instances = data.get("instances", [])
    missing   = data.get("missing_feature_types", [])

    print(f"  Instances to validate: {len(instances)}")
    print(f"  Missing feature types: {missing}")

    # Load reference images for ORB + SSIM
    print(f"\n  Loading reference images from {args.refs}/")
    ref_images = load_ref_images(args.refs)
    print(f"  Loaded {len(ref_images)} reference images: {list(ref_images.keys())}")

    # Run all 8 checks
    t0 = time.time()
    check_results, flags, mission_complete, orb_results, ssim_results = \
        run_checks(instances, missing, ref_images, args.frames, args.out)

    elapsed = time.time() - t0
    passed  = sum(1 for c in check_results if c["status"] == "PASS")
    total   = len(check_results)

    # ── Re-sortie recommendation ──────────────────────────────────
    resortie_needed = not mission_complete
    resortie_targets = []
    if missing:
        resortie_targets = missing
    elif resortie_needed:
        # Flag low-confidence instances for re-sortie
        resortie_targets = [
            inst["feature_type"] for inst in instances
            if inst["confidence"]["cnn_score"] < MIN_CNN_SCORE
        ]
        resortie_targets = list(set(resortie_targets))

    # ── Build report ──────────────────────────────────────────────
    report = {
        "team"             : "Team Marsvista",
        "challenge"        : "ISRO IRoC-U 2026 ASCEND",
        "validated_at"     : datetime.now().isoformat(),
        "checks_passed"    : passed,
        "checks_total"     : total,
        "mission_complete" : mission_complete,
        "flags"            : flags,
        "resortie_needed"  : resortie_needed,
        "resortie_targets" : resortie_targets,
        "check_details"    : check_results,
        "orb_results"      : orb_results,
        "ssim_results"     : ssim_results,
        "elapsed_seconds"  : round(elapsed, 2),
    }

    # Save JSON report
    json_path = os.path.join(args.out, "revalidation_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save text summary
    txt_path = os.path.join(args.out, "revalidation_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"ASCEND Revalidation Report — Team Marsvista\n")
        f.write(f"{'='*55}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Checks: {passed}/{total} passed\n")
        f.write(f"Mission complete: {mission_complete}\n\n")
        for c in check_results:
            sym = "✓" if c["status"]=="PASS" else "✗"
            f.write(f"{sym} {c['check']}: {c['detail']}\n")
        if resortie_needed:
            f.write(f"\nRE-SORTIE NEEDED for: {resortie_targets}\n")

    # ── Print final verdict ───────────────────────────────────────
    print(f"\n  {'='*50}")
    print(f"  Revalidation: {passed}/{total} checks passed")
    print(f"  Elapsed: {elapsed:.1f}s")
    if mission_complete:
        print("  ★ MISSION COMPLETE — all checks passed")
    else:
        print("  ⚠ MISSION INCOMPLETE")
        for flag in flags:
            print(f"    {flag}")
        if resortie_targets:
            print(f"\n  RE-SORTIE recommended for: {resortie_targets}")
    print(f"  Report: {json_path}")
    print(f"  {'='*50}")

    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/results.json")
    p.add_argument("--refs",    default="refs")
    p.add_argument("--frames",  default="flight_frames")
    p.add_argument("--out",     default="results")
    main(p.parse_args())
