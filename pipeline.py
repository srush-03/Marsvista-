"""
pipeline.py — ISRO IRoC-U 2026 ASCEND | Team Marsvista
=======================================================
Autonomous post-flight pipeline. Run once after drone lands.

SYSTEM BOUNDARY — THIS CODE PRODUCES:
    results/mosaic.jpg               arena mosaic image
    results/results.json             all detected instances + coordinates
    results/revalidation_report.json 8-check verification status
    results/proof_*.jpg              HD proof image per instance
    results/cmp_*.jpg                128x128 comparison per instance
    results/handoff/                 clean package for dashboard + telemetry teams

PIPELINE:
    STITCH → MATCH → REVALIDATE → HANDOFF

    Step 1 — Stitching
        All saved HD frames → single arena mosaic
        SLAM pose guides frame placement on canvas
        ORB refines alignment between consecutive frames

    Step 2 — Matching
        Each HD frame scanned with multi-scale sliding windows
        Each window → 128x128 LR in memory (never written to disk)
        DINOv2 + LBP + adaptive HSV fusion score vs 3 LR references
        Detections clustered by 1m radius → instances

    Step 3 — Revalidation
        8 independent checks using ORB + SSIM (not DINOv2)
        Determines: MISSION COMPLETE or re-sortie needed

    Step 4 — Handoff
        Packages results for dashboard team and telemetry team
        Writes handoff manifest listing all output files

Usage:
    python pipeline.py
    python pipeline.py --skip-stitch
    python pipeline.py --threshold 0.48
    python pipeline.py --frames flight_frames/ --refs refs/ --out results/
"""

import os
import sys
import json
import time
import re
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────
REFS_DIR     = "refs"
FRAMES_DIR   = "flight_frames"
OUT_DIR      = "results"
MATCH_THRESH = 0.52
STITCH_EVERY = 2


# ── UTILITIES ─────────────────────────────────────────────────────
def banner(n, title):
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {title}")
    print("="*60)


def load_module(name, filename):
    """Dynamically load a sibling .py file as a module."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} not found at {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── PREREQUISITES ─────────────────────────────────────────────────
def check_prerequisites(frames_dir, refs_dir):
    """
    Verify all required inputs exist before starting.
    Returns (errors, warnings) lists.
    """
    errors   = []
    warnings = []
    _RE = re.compile(r"frame_\d+_x-?[\d.]+_y-?[\d.]+_z-?[\d.]+")

    # Check frames
    if not os.path.isdir(frames_dir):
        errors.append(f"Frames directory not found: {frames_dir}/")
    else:
        all_frames = list(Path(frames_dir).glob("*.jpg"))
        if not all_frames:
            errors.append(f"No .jpg files in {frames_dir}/")
        else:
            posed = [f for f in all_frames if _RE.search(f.name)]
            if not posed:
                errors.append(
                    "Frames found but none have pose in filename. "
                    "Was ORB-SLAM3 running during flight?")
            else:
                # Warn if >90% frames have zero pose (SLAM failure)
                zero = sum(1 for f in posed
                           if "_x0.000_y0.000_z0.000" in f.name)
                if zero > len(posed) * 0.9:
                    warnings.append(
                        f"{zero}/{len(posed)} frames have pose (0,0,0). "
                        "SLAM may have failed — coordinates will be wrong.")

    # Check refs
    if not os.path.isdir(refs_dir):
        errors.append(f"References directory not found: {refs_dir}/")
    else:
        lr_refs = [f for f in os.listdir(refs_dir)
                   if f.lower().endswith((".jpg",".png"))
                   and "HD" not in f and "metadata" not in f]
        if not lr_refs:
            errors.append(
                f"No LR reference images in {refs_dir}/. "
                "Run reference_creator.py first.")

    return errors, warnings


# ── STEP 1: STITCH ────────────────────────────────────────────────
def run_stitch(frames_dir, out_dir, every_n):
    """Build arena mosaic from HD frames."""
    banner(1, "STITCHING — building arena mosaic")
    print("  All saved HD frames → single top-down mosaic image.")
    print("  SLAM pose guides frame placement. ORB refines alignment.")

    t0 = time.time()
    try:
        stitch = load_module("stitch", "stitch.py")
    except FileNotFoundError as e:
        print(f"  WARNING: {e} — skipping stitching.")
        return None

    import argparse as _ap
    args = _ap.Namespace(
        frames  = frames_dir,
        out     = out_dir,
        results = os.path.join(out_dir, "results.json"),
        every   = every_n,
        scale   = 0.5,
    )
    try:
        stitch.main(args)
        path = os.path.join(out_dir, "mosaic.jpg")
        if os.path.exists(path):
            size_kb = os.path.getsize(path) // 1024
            print(f"\n  ✓ Mosaic: {path} ({size_kb} KB) [{time.time()-t0:.1f}s]")
            return path
    except Exception as e:
        print(f"  WARNING: Stitching failed — {e}")
        print("  Continuing to matching step.")
    return None


# ── STEP 2: MATCH ─────────────────────────────────────────────────
def run_match(frames_dir, refs_dir, out_dir, threshold):
    """Match HD frames against LR references."""
    banner(2, "MATCHING — detecting feature instances")
    print("  Multi-scale sliding windows → 128x128 LR in memory → fused score.")
    print("  Clusters detections by 1m radius → one entry per physical instance.")

    t0 = time.time()
    try:
        pm = load_module("post_match", "post_match.py")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None

    import argparse as _ap
    args = _ap.Namespace(
        refs      = refs_dir,
        frames    = frames_dir,
        out       = out_dir,
        threshold = threshold,
        skip_blur = False,
    )
    try:
        pm.main(args)
        path = os.path.join(out_dir, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            n = data.get("total_instances", 0)
            print(f"\n  ✓ Matching done: {n} instances found [{time.time()-t0:.1f}s]")
            return path
    except Exception as e:
        print(f"  ERROR: Matching failed — {e}")
        import traceback; traceback.print_exc()
    return None


# ── STEP 3: REVALIDATE ────────────────────────────────────────────
def run_revalidation(results_path, refs_dir, frames_dir, out_dir):
    """Independent 8-check revalidation using ORB + SSIM."""
    banner(3, "REVALIDATION — independent verification (ORB + SSIM)")
    print("  8 checks using ORB keypoints and SSIM — different from DINOv2.")
    print("  Independently confirms or flags each detected instance.")

    t0 = time.time()
    try:
        rv = load_module("revalidation", "revalidation.py")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None

    import argparse as _ap
    args = _ap.Namespace(
        results = results_path,
        refs    = refs_dir,
        frames  = frames_dir,
        out     = out_dir,
    )
    try:
        report = rv.main(args)
        if report:
            passed = report.get("checks_passed", 0)
            total  = report.get("checks_total", 0)
            print(f"\n  ✓ Revalidation done: {passed}/{total} checks [{time.time()-t0:.1f}s]")
        return report
    except Exception as e:
        print(f"  ERROR: Revalidation failed — {e}")
        import traceback; traceback.print_exc()
    return None


# ── STEP 4: HANDOFF ───────────────────────────────────────────────
def run_handoff(results_path, reval_path, mosaic_path, out_dir):
    """
    Package all outputs for dashboard team and telemetry team.
    Writes a manifest listing every file they need to consume.
    """
    banner(4, "HANDOFF — packaging outputs for other teams")

    handoff_dir = os.path.join(out_dir, "handoff")
    os.makedirs(handoff_dir, exist_ok=True)

    # Collect all output files
    files = {}

    # Core data files
    for fname, label in [
        ("results.json",             "detection_results"),
        ("revalidation_report.json", "revalidation_status"),
        ("mosaic.jpg",               "arena_mosaic"),
        ("mosaic_annotated.jpg",     "arena_mosaic_annotated"),
        ("coordinate_map.json",      "coordinate_reference"),
    ]:
        src = os.path.join(out_dir, fname)
        if os.path.exists(src):
            files[label] = {
                "file"       : fname,
                "path"       : src,
                "size_bytes" : os.path.getsize(src),
            }

    # Proof images
    proofs = sorted(Path(out_dir).glob("proof_*.jpg"))
    if proofs:
        files["proof_images"] = [
            {"file": p.name, "path": str(p), "size_bytes": p.stat().st_size}
            for p in proofs
        ]

    # Comparison images
    cmps = sorted(Path(out_dir).glob("cmp_*.jpg"))
    if cmps:
        files["comparison_images"] = [
            {"file": p.name, "path": str(p), "size_bytes": p.stat().st_size}
            for p in cmps
        ]

    # Load results summary for manifest
    mission_complete = False
    resortie_needed  = False
    resortie_targets = []
    instances_summary = []

    if results_path and os.path.exists(results_path):
        with open(results_path) as f:
            res = json.load(f)
        for inst in res.get("instances", []):
            instances_summary.append({
                "instance_id"  : inst["instance_id"],
                "feature_type" : inst["feature_type"],
                "coordinates"  : inst["coordinates"],
                "cnn_score"    : inst["confidence"]["cnn_score"],
                "proof_image"  : inst.get("hd_proof_image", ""),
            })

    if reval_path and os.path.exists(reval_path):
        with open(reval_path) as f:
            rv = json.load(f)
        mission_complete = rv.get("mission_complete", False)
        resortie_needed  = rv.get("resortie_needed", False)
        resortie_targets = rv.get("resortie_targets", [])

    # Build manifest
    manifest = {
        "team"             : "Team Marsvista",
        "challenge"        : "ISRO IRoC-U 2026 ASCEND",
        "generated_at"     : datetime.now().isoformat(),
        "mission_complete" : mission_complete,
        "resortie_needed"  : resortie_needed,
        "resortie_targets" : resortie_targets,
        "total_instances"  : len(instances_summary),
        "instances"        : instances_summary,
        "files"            : files,
        "notes": {
            "results.json"            : "Primary output. All detected instances with coordinates and confidence scores.",
            "revalidation_report.json": "8-check verification. Read mission_complete and flags fields.",
            "mosaic.jpg"              : "Top-down arena map stitched from drone HD frames.",
            "proof_*.jpg"             : "HD proof image per instance. Bounding box drawn on matched region.",
            "cmp_*.jpg"               : "Side-by-side 128x128 comparison: LR reference (left) vs live crop (right).",
        }
    }

    manifest_path = os.path.join(handoff_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Handoff package ready: {handoff_dir}/")
    print(f"  Manifest: {manifest_path}")
    print(f"\n  Files for dashboard + telemetry teams:")
    for label, info in files.items():
        if isinstance(info, list):
            print(f"    {label}: {len(info)} files")
        else:
            kb = info["size_bytes"] // 1024
            print(f"    {info['file']:<40} {kb} KB")

    return manifest_path


# ── TERMINAL SUMMARY ──────────────────────────────────────────────
def print_final_summary(results_path, reval_report):
    """Print clean terminal summary — team's final output."""
    print(f"\n{'='*60}")
    print("  MISSION SUMMARY — Team Marsvista")
    print("="*60)

    if not results_path or not os.path.exists(results_path):
        print("  No results available.")
        return

    with open(results_path) as f:
        data = json.load(f)

    instances = data.get("instances", [])
    missing   = data.get("missing_feature_types", [])

    # Group by feature type
    by_type = {}
    for inst in instances:
        by_type.setdefault(inst["feature_type"], []).append(inst)

    print(f"\n  {'FEATURE TYPE':<35} {'INST':<6} {'BEST SCORE':<12} {'COORDINATES'}")
    print(f"  {'-'*70}")

    for ft, insts in by_type.items():
        for inst in insts:
            score = inst["confidence"]["cnn_score"]
            c     = inst["coordinates"]
            coord = f"x={c['x']}m  y={c['y']}m"
            print(f"  {inst['instance_id']:<35} #{inst['instance_index']:<4} "
                  f"{score:<12.3f} {coord}")

    for ft in missing:
        print(f"  {ft:<35} NONE  —  NOT FOUND")

    print(f"\n  Total instances detected : {len(instances)}")
    print(f"  Missing feature types    : {len(missing)}")

    if reval_report:
        p = reval_report.get("checks_passed", 0)
        t = reval_report.get("checks_total",  0)
        ok = reval_report.get("mission_complete", False)
        print(f"  Revalidation             : {p}/{t} checks passed")
        print(f"  Mission status           : {'★ COMPLETE' if ok else '⚠ INCOMPLETE — re-sortie needed'}")
        if reval_report.get("resortie_targets"):
            print(f"  Re-sortie targets        : {reval_report['resortie_targets']}")

    print(f"\n  Output folder: results/")
    print(f"  Handoff ready: results/handoff/manifest.json")
    print("="*60)


# ── MAIN ──────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="ASCEND autonomous post-flight pipeline — Team Marsvista")
    p.add_argument("--refs",         default=REFS_DIR)
    p.add_argument("--frames",       default=FRAMES_DIR)
    p.add_argument("--out",          default=OUT_DIR)
    p.add_argument("--threshold",    type=float, default=MATCH_THRESH)
    p.add_argument("--skip-stitch",  action="store_true",
                   help="Skip stitching step (faster, for re-runs)")
    p.add_argument("--stitch-every", type=int, default=STITCH_EVERY,
                   help="Use every Nth frame for stitching (default 2)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("\n" + "="*60)
    print("  ASCEND Post-Flight Pipeline — Team Marsvista")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Prerequisites
    print("\n  Checking inputs...")
    errors, warnings = check_prerequisites(args.frames, args.refs)
    for w in warnings:
        print(f"  ⚠ {w}")
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        print("\n  Cannot start — fix errors above first.")
        sys.exit(1)
    print("  ✓ All inputs ready.")

    t_total = time.time()

    # Step 1
    mosaic_path = None
    if not args.skip_stitch:
        mosaic_path = run_stitch(args.frames, args.out, args.stitch_every)
    else:
        print("\n  [Step 1 skipped — --skip-stitch]")

    # Step 2
    results_path = run_match(args.frames, args.refs, args.out, args.threshold)
    if not results_path:
        print("\n  FATAL: Matching failed — cannot revalidate or handoff.")
        sys.exit(1)

    # Step 3
    reval_report = run_revalidation(
        results_path, args.refs, args.frames, args.out)

    # Step 4 — handoff
    reval_json = os.path.join(args.out, "revalidation_report.json")
    run_handoff(results_path, reval_json, mosaic_path, args.out)

    # Terminal summary
    print_final_summary(results_path, reval_report)

    print(f"\n  Total time: {time.time()-t_total:.1f}s")
    print(f"  Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
