"""
reference_creator.py — ISRO IRoC-U 2026 ASCEND | Team Marsvista
================================================================
Run ONCE before flight at arena.

TWO MODES:
    Mode A (Elimination round): You take the photos yourself
        Point camera top-down at each feature from ~1.5m height
        Press C to capture, S to save, Q when done

    Mode B (Final round): ISRO provides reference images
        Place their images in refs/ folder
        Run with --validate flag to check quality

Usage:
    python reference_creator.py              # interactive webcam mode
    python reference_creator.py --validate   # validate existing refs/
    python reference_creator.py --source img.jpg --name ref_0_rock  # from file

Output:
    refs/ref_0_rock_HD.jpg      <- original HD capture
    refs/ref_0_rock_LR.jpg      <- 128x128 downsampled (used for matching)
    refs/metadata.json          <- quality scores, capture info
"""

import cv2
import numpy as np
import os
import json
import argparse
import time
from datetime import datetime
from skimage.metrics import structural_similarity as ssim_fn

# ── CONFIG ────────────────────────────────────────────────────────
REFS_DIR     = "refs"
LR_SIZE      = (128, 128)
MIN_SHARPNESS = 50.0     # Laplacian variance — warn below this, not hard reject
MIN_CONTRAST  = 15.0     # std of grayscale — warn below this


# ── QUALITY CHECK ─────────────────────────────────────────────────
def quality_check(img_bgr, name="image"):
    """
    Returns (passed: bool, report: dict)
    Checks sharpness, contrast, and whether image is mostly background.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var   = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    contrast  = float(gray.std())
    mean_v    = float(gray.mean())

    issues = []
    if lap_var < MIN_SHARPNESS:
        issues.append(f"BLURRY (sharpness={lap_var:.1f}, need >{MIN_SHARPNESS})")
    if contrast < MIN_CONTRAST:
        issues.append(f"LOW CONTRAST (std={contrast:.1f}, need >{MIN_CONTRAST})")
    if mean_v < 30 or mean_v > 225:
        issues.append(f"BAD EXPOSURE (mean brightness={mean_v:.1f})")

    report = {
        "name"      : name,
        "sharpness" : round(lap_var, 2),
        "contrast"  : round(contrast, 2),
        "brightness": round(mean_v, 2),
        "issues"    : issues,
        "passed"    : len(issues) == 0,
    }
    return report["passed"], report


# ── MULTI-METHOD DOWNSAMPLE ───────────────────────────────────────
def make_lr(img_bgr):
    """
    Rulebook allows multiple downsampling methods.
    We average INTER_AREA + INTER_LINEAR for best quality at 128x128.
    INTER_AREA is best for downscaling (anti-aliasing).
    """
    area   = cv2.resize(img_bgr, LR_SIZE, interpolation=cv2.INTER_AREA)
    linear = cv2.resize(img_bgr, LR_SIZE, interpolation=cv2.INTER_LINEAR)
    lr     = cv2.addWeighted(area, 0.7, linear, 0.3, 0)

    # Apply mild CLAHE to enhance local contrast at 128x128
    lab  = cv2.cvtColor(lr, cv2.COLOR_BGR2LAB)
    cl   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    lr   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return lr


# ── INTERACTIVE CAPTURE MODE ──────────────────────────────────────
def interactive_mode():
    os.makedirs(REFS_DIR, exist_ok=True)
    metadata = []

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    feature_names = [
        "ref_0_rock",    # change these names to match your feature types
        "ref_1_oxide",   # names are used for automatic profile selection
        "ref_2_ice",     # keywords: rock/layer/stone, oxide/rust/iron, ice/foil/reflect
    ]
    current_idx   = 0
    captured      = {}   # name -> (hd, lr) image pair
    preview_lr    = None

    print("\n" + "="*55)
    print("  REFERENCE CREATOR — Interactive Mode")
    print("="*55)
    print("  C = capture current frame")
    print("  S = save captured frame as reference")
    print("  R = retake (discard current capture)")
    print("  N = next feature")
    print("  Q = quit and save all")
    print("="*55 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        display = frame.copy()
        h, w    = display.shape[:2]

        if current_idx >= len(feature_names):
            cv2.putText(display, "All 3 references captured! Press Q to save.",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            fname = feature_names[current_idx]
            done  = sum(1 for n in feature_names[:current_idx] if n in captured)

            # Status bar
            cv2.rectangle(display, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(display,
                f"Feature {current_idx+1}/3: {fname}   |   Done: {done}/3",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 255, 0) if fname in captured else (0, 200, 255), 2)

            # Quality meter
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            bar_col   = (0, 255, 0) if sharpness > MIN_SHARPNESS else (0, 100, 255)
            bar_w     = min(int(sharpness / 5), w - 20)
            cv2.rectangle(display, (10, h-25), (10+bar_w, h-10), bar_col, -1)
            cv2.putText(display,
                f"Sharpness: {sharpness:.0f} ({'OK' if sharpness>MIN_SHARPNESS else 'BLURRY — move camera'})",
                (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_col, 1)

            # Instructions
            instr = "C=capture" if fname not in captured else "S=save  R=retake  N=next"
            cv2.putText(display, instr,
                (w-200, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Show LR preview if captured
            if preview_lr is not None:
                # Upscale for visibility
                prev = cv2.resize(preview_lr, (256, 256), interpolation=cv2.INTER_NEAREST)
                display[65:65+256, w-266:w-10] = prev
                cv2.rectangle(display, (w-266, 65), (w-10, 321), (0,200,255), 2)
                cv2.putText(display, "128x128 LR preview",
                    (w-260, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)

        cv2.imshow("Reference Creator — Team Marsvista", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c') and current_idx < len(feature_names):
            fname = feature_names[current_idx]
            passed, report = quality_check(frame, fname)
            if not passed:
                print(f"\n  QUALITY WARNING for {fname}:")
                for issue in report["issues"]:
                    print(f"    - {issue}")
                print("  Hold still, ensure good lighting, try again.")
            else:
                lr = make_lr(frame)
                preview_lr = lr
                captured[fname] = (frame.copy(), lr)
                print(f"\n  Captured: {fname}")
                print(f"    Sharpness: {report['sharpness']}  Contrast: {report['contrast']}")
                print("  Press S to save, R to retake.")

        elif key == ord('s') and current_idx < len(feature_names):
            fname = feature_names[current_idx]
            if fname not in captured:
                print("  Nothing captured yet — press C first.")
            else:
                hd, lr = captured[fname]
                hd_path = os.path.join(REFS_DIR, f"{fname}_HD.jpg")
                lr_path = os.path.join(REFS_DIR, f"{fname}_LR.jpg")
                cv2.imwrite(hd_path, hd, [cv2.IMWRITE_JPEG_QUALITY, 98])
                cv2.imwrite(lr_path, lr)
                # Quality check on HD image (not LR — sharpness drops after downsample)
                _, report = quality_check(hd, fname)
                report["hd_path"] = hd_path
                report["lr_path"] = lr_path
                report["captured_at"] = datetime.now().isoformat()
                metadata.append(report)
                print(f"  Saved: {hd_path}")
                print(f"  Saved: {lr_path}")
                current_idx += 1
                preview_lr = None

        elif key == ord('r') and current_idx < len(feature_names):
            fname = feature_names[current_idx]
            if fname in captured:
                del captured[fname]
                preview_lr = None
                print(f"  Discarded capture for {fname} — recapture with C.")

        elif key == ord('n') and current_idx < len(feature_names):
            fname = feature_names[current_idx]
            if fname not in captured:
                print(f"  WARNING: {fname} not captured — skipping anyway.")
            current_idx += 1
            preview_lr = None

    cap.release()
    cv2.destroyAllWindows()

    # Save metadata
    meta_path = os.path.join(REFS_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "team"       : "Team Marsvista",
            "created_at" : datetime.now().isoformat(),
            "lr_size"    : list(LR_SIZE),
            "references" : metadata,
        }, f, indent=2)

    print(f"\n  Metadata saved: {meta_path}")
    print(f"  {len(metadata)} references ready.")


# ── VALIDATE EXISTING REFS ────────────────────────────────────────
def validate_refs():
    """Check quality of all existing LR refs in refs/ folder."""
    lr_files = sorted([
        f for f in os.listdir(REFS_DIR)
        if f.endswith("_LR.jpg") or f.endswith("_LR.png")
    ])

    if not lr_files:
        print(f"No LR reference files found in {REFS_DIR}/")
        print("Expected filenames ending in _LR.jpg")
        return

    print(f"\nValidating {len(lr_files)} reference images...\n")
    all_passed = True

    for fname in lr_files:
        path = os.path.join(REFS_DIR, fname)
        img  = cv2.imread(path)
        if img is None:
            print(f"  ERROR: Cannot read {fname}")
            all_passed = False
            continue

        # Check size
        h, w = img.shape[:2]
        if (w, h) != LR_SIZE:
            print(f"  WARNING: {fname} is {w}x{h}, expected 128x128 — resizing.")
            img = cv2.resize(img, LR_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(path, img)

        passed, report = quality_check(img, fname)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {fname}")
        print(f"    Sharpness={report['sharpness']}  "
              f"Contrast={report['contrast']}  "
              f"Brightness={report['brightness']}")
        if report["issues"]:
            for issue in report["issues"]:
                print(f"    - {issue}")
            all_passed = False

    # Check similarity between refs — they should be DIFFERENT
    imgs = []
    for fname in lr_files:
        img = cv2.imread(os.path.join(REFS_DIR, fname))
        if img is not None:
            imgs.append((fname, img))

    if len(imgs) >= 2:
        print("\n  Similarity check (refs should be DIFFERENT from each other):")
        for i in range(len(imgs)):
            for j in range(i+1, len(imgs)):
                n1, i1 = imgs[i]
                n2, i2 = imgs[j]
                g1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
                score = ssim_fn(g1, g2, data_range=255)
                flag  = "WARNING — TOO SIMILAR" if score > 0.7 else "OK"
                print(f"    {n1} vs {n2}: SSIM={score:.3f}  [{flag}]")
                if score > 0.7:
                    all_passed = False

    print(f"\n  Overall: {'ALL PASS' if all_passed else 'ISSUES FOUND — fix before flight'}")


# ── FROM FILE MODE ────────────────────────────────────────────────
def from_file(source_path, name):
    """Import an existing image as a reference."""
    os.makedirs(REFS_DIR, exist_ok=True)
    img = cv2.imread(source_path)
    if img is None:
        print(f"ERROR: Cannot read {source_path}")
        return

    # Resize to 1280x720 if needed
    h, w = img.shape[:2]
    if w < 1280 or h < 720:
        print(f"WARNING: Image is {w}x{h} — smaller than 1280x720 HD requirement.")
    hd = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR) if (w!=1280 or h!=720) else img

    lr = make_lr(hd)
    passed, report = quality_check(lr, name)

    hd_path = os.path.join(REFS_DIR, f"{name}_HD.jpg")
    lr_path = os.path.join(REFS_DIR, f"{name}_LR.jpg")
    cv2.imwrite(hd_path, hd, [cv2.IMWRITE_JPEG_QUALITY, 98])
    cv2.imwrite(lr_path, lr)

    print(f"Saved HD: {hd_path}")
    print(f"Saved LR: {lr_path}")
    print(f"Quality: {'PASS' if passed else 'ISSUES: ' + str(report['issues'])}")


# ── ENTRY POINT ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing refs in refs/ folder")
    parser.add_argument("--source",   type=str, default=None,
                        help="Import an existing image file as reference")
    parser.add_argument("--name",     type=str, default="ref_custom",
                        help="Name for imported reference (used with --source)")
    args = parser.parse_args()

    if args.validate:
        validate_refs()
    elif args.source:
        from_file(args.source, args.name)
    else:
        interactive_mode()
