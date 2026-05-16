"""
post_match.py v2 — ISRO IRoC-U 2026 ASCEND | Team Marsvista
=============================================================
Finds ALL instances of ALL feature types across the arena.

KEY UPGRADES:
    1. Multi-scale sliding windows (512, 256, 128px)
    2. Multi-prototype matching (fires if ANY prototype matches)
    3. Proper multi-instance spatial clustering (1m radius)
    4. Blur + background rejection filters
    5. Detailed diagnostic output when matches are missing

Usage:
    python post_match.py
    python post_match.py --threshold 0.48   # more sensitive
    python post_match.py --skip-blur        # include blurry frames
"""

import cv2
import numpy as np
import os, json, re, time, argparse, warnings
from datetime import datetime
from pathlib import Path
from skimage.metrics import structural_similarity as ssim_fn

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────
REFS_DIR       = "refs"
FRAMES_DIR     = "flight_frames"
OUT_DIR        = "results"
LR_SIZE        = (128, 128)
WINDOW_SCALES  = [512, 256, 128]
SSIM_THRESH    = 0.30   # raised — still permissive but filters truly unrelated windows
CNN_THRESH     = 0.52
CLUSTER_DIST_M = 1.0
MAX_INSTANCES  = 3
MIN_SHARPNESS  = 50.0

_RE = re.compile(r"frame_(\d+)_x(-?[\d.]+)_y(-?[\d.]+)_z(-?[\d.]+)")

def parse_pose(fname):
    m = _RE.search(os.path.basename(str(fname)))
    return (int(m[1]), float(m[2]), float(m[3]), float(m[4])) if m else None

def to_lr(img):
    a = cv2.resize(img, LR_SIZE, interpolation=cv2.INTER_AREA)
    l = cv2.resize(img, LR_SIZE, interpolation=cv2.INTER_LINEAR)
    r = cv2.addWeighted(a, 0.7, l, 0.3, 0)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2LAB)
    cl  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    lab[:,:,0] = cl.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def load_background_profile(refs_dir):
    """Load background HSV stats from metadata.json if present."""
    meta_path = os.path.join(refs_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        bg = meta.get("background") or meta.get("bg_hsv")
        if bg:
            print(f"  Background profile: H={bg.get('h_mean',0):.1f} "
                  f"S={bg.get('s_mean',0):.1f} V={bg.get('v_mean',0):.1f}")
            return bg
    except Exception:
        pass
    return None

def is_background(crop, bg_stats=None):
    """
    Arena-adaptive background rejection.
    Always rejects sensor artifacts (pure black/white).
    With bg_stats: also rejects windows statistically
    indistinguishable from background soil — works for ANY soil color.
    """
    hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    mean_v = float(hsv[:,:,2].mean())
    if mean_v > 245 or mean_v < 10:
        return True   # sensor artifact

    if bg_stats is not None:
        mean_h   = float(hsv[:,:,0].mean())
        mean_s   = float(hsv[:,:,1].mean())
        bg_s_std = max(bg_stats.get('s_std', 30), 5.0)
        bg_v_std = max(bg_stats.get('v_std', 30), 5.0)
        dh = abs(mean_h   - bg_stats.get('h_mean', 0)) / 180.0
        ds = abs(mean_s   - bg_stats.get('s_mean', 0)) / bg_s_std
        dv = abs(mean_v   - bg_stats.get('v_mean', 0)) / bg_v_std
        if dh < 0.12 and ds < 1.2 and dv < 1.2:
            return True   # indistinguishable from soil
    return False

def sliding_windows(frame, bg_stats=None):
    h, w = frame.shape[:2]
    yield to_lr(frame), 0, 0, w, h, "full"
    for scale in WINDOW_SCALES:
        if scale > min(h, w): continue
        stride = max(64, scale // 4)
        ys = list(range(0, h-scale+1, stride)) + [h-scale]
        xs = list(range(0, w-scale+1, stride)) + [w-scale]
        seen = set()
        for y in ys:
            for x in xs:
                key = (x, y)
                if key in seen: continue
                seen.add(key)
                crop = frame[y:y+scale, x:x+scale]
                if is_background(crop, bg_stats): continue
                yield to_lr(crop), x, y, x+scale, y+scale, scale


class Extractor:
    def __init__(self):
        import torch; self.torch = torch
        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2","dinov2_vits14",
                pretrained=True, verbose=False)
            self.model.eval()
            self.name = "DINOv2 ViT-S/14"
            import torchvision.transforms as T
            self.tf = T.Compose([T.ToPILImage(),T.Resize((224,224)),
                T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
            print(f"  Extractor: {self.name}")
        except Exception as e:
            print(f"  DINOv2 failed ({type(e).__name__}) → MobileNetV2")
            from torchvision.models import mobilenet_v2,MobileNet_V2_Weights
            import torchvision.transforms as T, torch.nn as nn
            m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            self.model = nn.Sequential(*list(m.children())[:-1])
            self.model.eval()
            self.name = "MobileNetV2"
            import torchvision.transforms as T
            self.tf = T.Compose([T.ToPILImage(),T.Resize((224,224)),
                T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])

    def vec(self, img_bgr):
        rgb = img_bgr[:,:,::-1].copy()
        t   = self.tf(rgb).unsqueeze(0)
        with self.torch.no_grad():
            f = self.model(t).squeeze().numpy().flatten()
        f /= (np.linalg.norm(f) + 1e-8)
        return f


def load_refs(refs_dir, ext):
    """
    Load all *_LR*.jpg files, group by feature name, build multi-prototype.
    """
    files = sorted([f for f in os.listdir(refs_dir)
        if f.lower().endswith((".jpg",".png"))
        and "HD" not in f and "metadata" not in f])
    if not files:
        return []

    # Group: strip _A/_B/_C/_LR suffixes
    groups = {}
    for f in files:
        base = re.sub(r"(_LR)?(_[A-C])?(\.(jpg|png))$","",f,flags=re.I)
        groups.setdefault(base,[]).append(f)

    refs = []
    for base, flist in groups.items():
        imgs = []
        for f in flist:
            img = cv2.imread(os.path.join(refs_dir,f))
            if img is not None:
                imgs.append(cv2.resize(img,LR_SIZE,interpolation=cv2.INTER_AREA))
        if not imgs: continue
        protos = [ext.vec(img) for img in imgs]
        mean_p = np.mean(protos,axis=0); mean_p /= np.linalg.norm(mean_p)+1e-8
        protos.append(mean_p)
        gray = cv2.cvtColor(imgs[0],cv2.COLOR_BGR2GRAY).astype(np.float32)
        refs.append({"name":base,"img":imgs[0],"gray":gray,
                     "files":flist,"protos":protos})
        print(f"  Ref: {base}  ({len(imgs)} capture(s), {len(protos)} prototypes)")
    return refs


def cluster(dets, dist_m=CLUSTER_DIST_M):
    if not dets: return []
    used = [False]*len(dets); clusters = []
    for i,d in enumerate(dets):
        if used[i]: continue
        grp=[d]; used[i]=True
        for j,d2 in enumerate(dets):
            if used[j]: continue
            if ((d["x"]-d2["x"])**2+(d["y"]-d2["y"])**2)**.5 < dist_m:
                grp.append(d2); used[j]=True
        best = max(grp,key=lambda c:c["cnn_score"])
        clusters.append({
            "best":best,"hits":len(grp),
            "mx":round(np.mean([c["x"] for c in grp]),3),
            "my":round(np.mean([c["y"] for c in grp]),3),
            "ms":round(np.mean([c["cnn_score"] for c in grp]),3),
        })
    clusters.sort(key=lambda c:c["hits"],reverse=True)
    return clusters


def main(args):
    os.makedirs(args.out, exist_ok=True)
    print("\n"+"="*60)
    print("  Post-Flight Multi-Instance Matcher v2 — Team Marsvista")
    print("="*60)

    frame_files = sorted([f for f in Path(args.frames).glob("*.jpg")
        if parse_pose(f.name)])
    if not frame_files:
        print(f"ERROR: No valid frames in {args.frames}"); return
    print(f"  Frames: {len(frame_files)}")

    print("\n  Loading extractor...")
    ext = Extractor()

    print(f"\n  Loading references from {args.refs}/")
    refs = load_refs(args.refs, ext)
    if not refs:
        print(f"ERROR: No references found in {args.refs}"); return
    print(f"  {len(refs)} feature type(s) ready.")

    # Load background profile for adaptive rejection
    bg_stats = load_background_profile(args.refs)
    if bg_stats is None:
        print("  Background profile: not found — using sensor-artifact filter only")
        print("  TIP: run reference_creator.py to save background profile")

    # ── Init FusionMatcher (DINOv2 + LBP + adaptive HSV) ─────────
    try:
        from feature_matcher import FusionMatcher
        fm = FusionMatcher(ext)
        fm.load_refs_from_dir(refs)
        if bg_stats:
            fm.set_background(bg_stats)
        use_fusion = True
        print("  Using: DINOv2 + LBP + adaptive HSV fusion\n")
    except ImportError:
        fm = None
        use_fusion = False
        print("  feature_matcher.py not found — DINOv2 only\n")

    raw = {r["name"]:[] for r in refs}
    t0=time.time(); skipped=corrupt=ssim_n=cnn_n=win_n=0

    for fi, fpath in enumerate(frame_files):
        pose = parse_pose(fpath.name)
        if not pose: continue
        idx,fx,fy,fz = pose

        frame = cv2.imread(str(fpath))
        if frame is None: corrupt+=1; continue
        h,w=frame.shape[:2]
        if w<1280 or h<720: frame=cv2.resize(frame,(1280,720))

        # Blur filter
        if not args.skip_blur:
            gray_f = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if float(cv2.Laplacian(gray_f,cv2.CV_32F).var()) < MIN_SHARPNESS:
                skipped+=1; continue

        if fi%300==0:
            print(f"  [{fi:5d}/{len(frame_files)}] "
                  f"SSIM={ssim_n} CNN={cnn_n} skip={skipped} "
                  f"t={time.time()-t0:.0f}s")

        for lr,wx1,wy1,wx2,wy2,scale in sliding_windows(frame, bg_stats):
            win_n+=1
            lg = cv2.cvtColor(lr,cv2.COLOR_BGR2GRAY).astype(np.float32)
            for r in refs:
                # SSIM pre-filter (fast reject)
                sc_ssim = ssim_fn(lg,r["gray"],data_range=255.0)
                if sc_ssim < SSIM_THRESH: continue
                ssim_n+=1

                # Fused score: DINOv2 + LBP + HSV
                if use_fusion:
                    win_vec = ext.vec(lr)
                    sc_fused, breakdown = fm.match(lr, r["name"],
                                                    precomputed_vec=win_vec)
                    sc_cnn = sc_fused
                else:
                    win_vec = ext.vec(lr)
                    sc_cnn  = max(float(np.dot(win_vec, p)) for p in r["protos"])
                    breakdown = {"fused": sc_cnn}

                if sc_cnn < args.threshold: continue
                cnn_n += 1
                sig = {k: val for k, val in breakdown.items()
                       if k in ("dinov2","lbp","hsv","fused","keyword")}
                raw[r["name"]].append({
                    "frame_file" : fpath.name, "frame_idx": idx,
                    "x": round(fx,4), "y": round(fy,4), "z": round(fz,4),
                    "window"     : [wx1,wy1,wx2,wy2], "scale": scale,
                    "ssim_score" : round(sc_ssim,4),
                    "cnn_score"  : round(sc_cnn,4),
                    "signal_breakdown": sig,
                })

    print(f"\n  Done in {time.time()-t0:.1f}s | "
          f"windows={win_n} ssim={ssim_n} cnn={cnn_n} "
          f"blur_skip={skipped} corrupt={corrupt}\n")

    # ── Cluster & save ────────────────────────────────────────────
    all_results=[]; missing=[]

    for r in refs:
        if not raw[r["name"]]:
            missing.append(r["name"])
            print(f"  ✗ {r['name']}: NO MATCH — check threshold or reference quality")
            continue

        clusters = cluster(raw[r["name"]])
        print(f"  ✓ {r['name']}: {len(clusters)} instance(s)")
        if len(clusters) > MAX_INSTANCES:
            print(f"    WARNING: {len(clusters)} instances > expected {MAX_INSTANCES} "
                  f"— consider raising --threshold")

        for ii, cl in enumerate(clusters):
            best   = cl["best"]
            coords = {"x":cl["mx"],"y":cl["my"],"z":round(best["z"],3)}

            # HD proof
            hd = cv2.imread(os.path.join(args.frames,best["frame_file"]))
            proof_name = cmp_name = ""
            if hd is not None:
                proof = hd.copy()
                x1,y1,x2,y2 = best["window"]
                col = (0,255,0) if best["cnn_score"]>=0.65 else (0,200,255)
                cv2.rectangle(proof,(x1,y1),(x2,y2),col,3)
                cv2.putText(proof,
                    f"{r['name']} #{ii} CNN={best['cnn_score']:.3f} hits={cl['hits']}",
                    (x1+5,max(y1-10,20)),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
                proof_name=(f"proof_{r['name']}_inst{ii}"
                            f"_fr{best['frame_idx']:06d}.jpg")
                cv2.imwrite(os.path.join(args.out,proof_name),proof,
                    [cv2.IMWRITE_JPEG_QUALITY,98])

                # LR comparison strip
                crop    = hd[y1:y2,x1:x2]
                lr_live = to_lr(crop)
                cmp     = np.zeros((128,268,3),dtype=np.uint8)
                cmp[:,:128]=r["img"]; cmp[:,140:]=lr_live
                cv2.line(cmp,(134,0),(134,128),(255,255,255),2)
                cv2.putText(cmp,"REF",(4,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,200,255),1)
                cv2.putText(cmp,"LIVE",(144,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
                cmp_name=(f"cmp_{r['name']}_inst{ii}"
                          f"_fr{best['frame_idx']:06d}.jpg")
                cv2.imwrite(os.path.join(args.out,cmp_name),
                    cv2.resize(cmp,(536,256)))

            all_results.append({
                "feature_type"   : r["name"],
                "instance_index" : ii,
                "instance_id"    : f"{r['name']}_inst{ii}",
                "coordinates"    : coords,
                "confidence"     : {
                    "cnn_score"  : best["cnn_score"],
                    "mean_score" : cl["ms"],
                    "hit_count"  : cl["hits"],
                    "scale_px"   : str(best["scale"]),
                },
                "frame_file"     : best["frame_file"],
                "frame_index"    : best["frame_idx"],
                "window_bbox_px" : best["window"],
                "hd_proof_image" : proof_name,
                "lr_comparison"  : cmp_name,
                "model"          : ext.name,
            })
            print(f"    inst#{ii}: x={coords['x']} y={coords['y']} "
                  f"CNN={best['cnn_score']:.3f} hits={cl['hits']}")

    # ── Write results.json ────────────────────────────────────────
    out_json = os.path.join(args.out,"results.json")
    with open(out_json,"w") as f:
        json.dump({
            "team":"Team Marsvista",
            "challenge":"ISRO IRoC-U 2026 ASCEND",
            "processed_at":datetime.now().isoformat(),
            "model":ext.name,
            "thresholds":{"ssim":SSIM_THRESH,"cnn":args.threshold,
                          "cluster_dist_m":CLUSTER_DIST_M},
            "stats":{"frames":len(frame_files),"blurry_skipped":skipped,
                     "corrupt":corrupt,"ssim_passed":ssim_n,"cnn_passed":cnn_n},
            "missing_feature_types":missing,
            "total_instances":len(all_results),
            "instances":all_results,
        },f,indent=2)

    print(f"\n{'='*60}")
    print(f"  Instances found: {len(all_results)} total")
    for r in refs:
        insts=[e for e in all_results if e["feature_type"]==r["name"]]
        print(f"  {r['name']:35s} {len(insts)} instance(s)")
    if missing:
        print(f"\n  MISSING: {missing} → trigger re-sortie")
    print(f"  Results: {out_json}")
    print("="*60)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--refs",      default=REFS_DIR)
    p.add_argument("--frames",    default=FRAMES_DIR)
    p.add_argument("--out",       default=OUT_DIR)
    p.add_argument("--threshold", type=float, default=CNN_THRESH)
    p.add_argument("--skip-blur", action="store_true")
    main(p.parse_args())
