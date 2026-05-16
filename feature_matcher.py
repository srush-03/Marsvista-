"""
feature_matcher.py — ISRO IRoC-U 2026 ASCEND | Team Marsvista
=============================================================
Arena-adaptive three-signal fusion matcher.

DESIGN PRINCIPLE — NO HARDCODED COLOR ASSUMPTIONS:
    Every signal is computed RELATIVE to the reference image itself.
    The system works regardless of whether the arena soil is:
    red, brown, gray, sandy, dark — any surface.

THREE SIGNALS:

    Signal 1 — DINOv2 / MobileNetV2 cosine similarity
        Compares deep feature vectors.
        Arena-independent by design (ImageNet pretrained).

    Signal 2 — Multi-radius LBP texture similarity
        Uses radius 1 AND radius 2 — captures both fine and
        medium-scale texture patterns.
        RELATIVE: compares histogram of window vs histogram of reference.
        Works on any surface because it measures texture DIFFERENCE,
        not absolute texture type.

    Signal 3 — HSV color deviation score
        NOT hardcoded to "red=oxide, white=ice".
        Instead: measures how much the window HSV stats MATCH
        the reference image HSV stats vs the background.
        Uses reference_creator's background profile if available.
        Falls back to pure reference-vs-window comparison.

FUSION WEIGHTS:
    Determined by analysing what signal is most reliable per type:
    oxide:   LBP heavy (fine texture vs soil is best signal)
    rock:    LBP + DINOv2 (texture pattern + shape)
    ice:     DINOv2 + HSV (appearance + relative brightness)
    default: balanced across all three

    BUT: if no keyword matches, uses balanced default.
    System works even if ISRO uses unexpected feature names.
"""

import cv2
import numpy as np
import re
from skimage.feature import local_binary_pattern

# ── LBP CONFIG ────────────────────────────────────────────────────
LBP_RADII  = [1, 2]       # multi-radius for coarse+fine texture
LBP_POINTS = 8
LBP_METHOD = "uniform"
LBP_BINS   = LBP_POINTS + 2   # uniform LBP = P+2 bins per radius

# ── FUSION PROFILES ───────────────────────────────────────────────
PROFILES = {
    "oxide" : {"dinov2": 0.30, "lbp": 0.50, "hsv": 0.20},
    "rust"  : {"dinov2": 0.30, "lbp": 0.50, "hsv": 0.20},
    "iron"  : {"dinov2": 0.30, "lbp": 0.50, "hsv": 0.20},
    "rock"  : {"dinov2": 0.45, "lbp": 0.55, "hsv": 0.00},
    "layer" : {"dinov2": 0.45, "lbp": 0.55, "hsv": 0.00},
    "stone" : {"dinov2": 0.45, "lbp": 0.55, "hsv": 0.00},
    "ice"   : {"dinov2": 0.65, "lbp": 0.10, "hsv": 0.25},
    "foil"  : {"dinov2": 0.65, "lbp": 0.10, "hsv": 0.25},
    "reflect": {"dinov2": 0.65, "lbp": 0.10, "hsv": 0.25},
    "bright": {"dinov2": 0.65, "lbp": 0.10, "hsv": 0.25},
    "default": {"dinov2": 0.50, "lbp": 0.35, "hsv": 0.15},
}

def _select_profile(name):
    nl = name.lower()
    for kw, p in PROFILES.items():
        if kw != "default" and kw in nl:
            return p, kw
    return PROFILES["default"], "default"


# ── MULTI-RADIUS LBP ──────────────────────────────────────────────
def compute_lbp_hist(img_bgr):
    """
    Compute concatenated multi-radius LBP histogram.
    Uses LAB L-channel for lighting robustness.
    Returns L2-normalised vector of shape (LBP_BINS * len(LBP_RADII),).
    """
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    gray = lab[:, :, 0].astype(np.uint8)
    hists = []
    for r in LBP_RADII:
        lbp  = local_binary_pattern(gray, LBP_POINTS, r, LBP_METHOD)
        h, _ = np.histogram(lbp.ravel(), bins=LBP_BINS,
                            range=(0, LBP_BINS))
        h = h.astype(np.float32)
        h /= (h.sum() + 1e-8)
        hists.append(h)
    combined = np.concatenate(hists)
    combined /= (np.linalg.norm(combined) + 1e-8)
    return combined


def lbp_similarity(hist_ref, hist_live):
    """
    Cosine similarity between two LBP histogram vectors.
    Both are already L2-normalised so dot product = cosine.
    Returns 0..1.
    """
    return float(np.dot(hist_ref, hist_live))


# ── ARENA-ADAPTIVE HSV SCORER ─────────────────────────────────────
def compute_hsv_stats(img_bgr):
    """Extract mean and std of each HSV channel from an image."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    stats = {}
    for i, ch in enumerate(['h','s','v']):
        stats[f'{ch}_mean'] = float(hsv[:,:,i].mean())
        stats[f'{ch}_std']  = float(hsv[:,:,i].std())
    return stats


def adaptive_hsv_score(ref_stats, win_stats, bg_stats=None):
    """
    Measures how similar the window HSV is to the reference,
    expressed RELATIVE to background.

    If background stats available:
        score = how much closer window is to ref than to background
    Else:
        score = direct similarity between ref and window HSV

    All comparisons are channel-wise normalised distances.
    Returns 0..1 (1 = window matches reference perfectly).
    """
    def channel_dist(a_stats, b_stats):
        # Weighted distance: saturation and value more important than hue
        dh = abs(a_stats['h_mean'] - b_stats['h_mean']) / 180.0
        ds = abs(a_stats['s_mean'] - b_stats['s_mean']) / 255.0
        dv = abs(a_stats['v_mean'] - b_stats['v_mean']) / 255.0
        return 0.25*dh + 0.45*ds + 0.30*dv

    dist_ref_win = channel_dist(ref_stats, win_stats)

    if bg_stats is not None:
        dist_ref_bg  = channel_dist(ref_stats, bg_stats)
        dist_bg_win  = channel_dist(bg_stats, win_stats)
        # Window is "more like ref than bg" → high score
        # Normalise: if window == ref → score 1; if window == bg → score 0
        denom = dist_ref_bg + 1e-6
        # How far is window from bg relative to ref-to-bg distance?
        score = 1.0 - (dist_ref_win / denom)
        return float(np.clip(score, 0.0, 1.0))
    else:
        # Direct: small distance = high score
        return float(np.exp(-dist_ref_win * 8.0))


# ── FUSION MATCHER ────────────────────────────────────────────────
class FusionMatcher:
    """
    Arena-adaptive fusion matcher.
    Works on any arena surface — no hardcoded color assumptions.

    Usage:
        fm = FusionMatcher(extractor)
        fm.load_refs_from_dir(refs_list)          # list from load_refs()
        fm.set_background(bg_hsv_stats)           # optional but recommended
        score, breakdown = fm.match(window_128, ref_name)
    """

    def __init__(self, extractor):
        self.extractor  = extractor
        self.refs       = {}
        self._bg_stats  = None   # set via set_background()

    # ── BACKGROUND REGISTRATION ───────────────────────────────────
    def set_background(self, bg_hsv_stats):
        """
        Register background HSV statistics for adaptive scoring.
        bg_hsv_stats: dict with h_mean, s_mean, v_mean, h_std, s_std, v_std
        Call this after loading the background profile from metadata.json
        or from reference_creator's calibration output.
        """
        self._bg_stats = bg_hsv_stats
        print(f"  FusionMatcher: background set "
              f"H={bg_hsv_stats.get('h_mean',0):.1f} "
              f"S={bg_hsv_stats.get('s_mean',0):.1f} "
              f"V={bg_hsv_stats.get('v_mean',0):.1f}")

    def load_bg_from_file(self, metadata_path):
        """
        Load background profile from reference_creator metadata.json
        or from a simple dict saved during capture.
        """
        import json, os
        if not os.path.exists(metadata_path):
            return
        with open(metadata_path) as f:
            meta = json.load(f)
        bg = meta.get("background", None)
        if bg:
            self.set_background(bg)
            return
        # Try webcam calibration format
        bg = meta.get("bg_hsv", None)
        if bg:
            self.set_background(bg)

    # ── REFERENCE LOADING ─────────────────────────────────────────
    def load_refs_from_dir(self, refs_list):
        """
        Load all refs from list returned by post_match.load_refs().
        refs_list: list of dicts with keys name, img, protos
        """
        for r in refs_list:
            self.load_ref(r["name"], r["img"], r["protos"])

    def load_ref(self, name, img_bgr, prototypes):
        """
        Register one reference.
        name:       string — used for profile selection
        img_bgr:    128x128 BGR reference image
        prototypes: list of L2-normalised DINOv2 vectors
        """
        profile, kw = _select_profile(name)
        lbp_hist    = compute_lbp_hist(img_bgr)
        hsv_stats   = compute_hsv_stats(img_bgr)

        self.refs[name] = {
            "img"      : img_bgr,
            "protos"   : prototypes,
            "lbp_hist" : lbp_hist,
            "hsv_stats": hsv_stats,
            "profile"  : profile,
            "keyword"  : kw,
        }
        print(f"  FusionMatcher ref '{name}' → profile={kw} "
              f"(DINOv2={profile['dinov2']} "
              f"LBP={profile['lbp']} "
              f"HSV={profile['hsv']})")

    # ── MATCH ─────────────────────────────────────────────────────
    def match(self, window_bgr, ref_name, precomputed_vec=None):
        """
        Compute fused similarity score.

        Returns:
            fused_score (float 0..1)
            breakdown   (dict with individual signal scores)
        """
        if ref_name not in self.refs:
            return 0.0, {"error": f"ref '{ref_name}' not loaded"}

        ref     = self.refs[ref_name]
        profile = ref["profile"]

        # ── Signal 1: DINOv2 ─────────────────────────────────────
        vec = precomputed_vec if precomputed_vec is not None \
              else self.extractor.vec(window_bgr)
        s1 = max(0.0, max(
            float(np.dot(vec, p)) for p in ref["protos"]
        ))

        # ── Signal 2: Multi-radius LBP ───────────────────────────
        if profile["lbp"] > 0:
            lbp_live = compute_lbp_hist(window_bgr)
            s2       = lbp_similarity(ref["lbp_hist"], lbp_live)
        else:
            s2 = 0.5

        # ── Signal 3: Adaptive HSV ───────────────────────────────
        if profile["hsv"] > 0:
            win_hsv = compute_hsv_stats(window_bgr)
            s3 = adaptive_hsv_score(
                ref["hsv_stats"], win_hsv, self._bg_stats)
        else:
            s3 = 0.5

        # ── Fusion ───────────────────────────────────────────────
        fused = (profile["dinov2"] * s1 +
                 profile["lbp"]    * s2 +
                 profile["hsv"]    * s3)

        breakdown = {
            "dinov2" : round(s1,    4),
            "lbp"    : round(s2,    4),
            "hsv"    : round(s3,    4),
            "fused"  : round(fused, 4),
            "keyword": ref["keyword"],
            "weights": profile,
        }
        return float(fused), breakdown


# ── STANDALONE TEST ───────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick test comparing two images.
    python feature_matcher.py <ref.jpg> <test.jpg> [background.jpg]
    """
    import sys, os, json

    if len(sys.argv) < 3:
        print("Usage: python feature_matcher.py <ref> <test> [background]")
        sys.exit(0)

    ref_img  = cv2.imread(sys.argv[1])
    test_img = cv2.imread(sys.argv[2])
    if ref_img is None or test_img is None:
        print("ERROR: cannot read images"); sys.exit(1)

    ref_img  = cv2.resize(ref_img,  (128,128), interpolation=cv2.INTER_AREA)
    test_img = cv2.resize(test_img, (128,128), interpolation=cv2.INTER_AREA)
    ref_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # Load extractor
    print("Loading extractor...")
    try:
        import torch
        model = torch.hub.load("facebookresearch/dinov2","dinov2_vits14",
            pretrained=True,verbose=False)
        model.eval()
        import torchvision.transforms as T
        tf = T.Compose([T.ToPILImage(),T.Resize((224,224)),T.ToTensor(),
            T.Normalize([.485,.456,.406],[.229,.224,.225])])
        def get_vec(img):
            rgb = img[:,:,::-1].copy()
            t   = tf(rgb).unsqueeze(0)
            with torch.no_grad():
                f = model(t).squeeze().numpy().flatten()
            f /= np.linalg.norm(f)+1e-8
            return f
        print("DINOv2 loaded")
    except Exception:
        from torchvision.models import mobilenet_v2,MobileNet_V2_Weights
        import torch, torch.nn as nn, torchvision.transforms as T
        m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        net = nn.Sequential(*list(m.children())[:-1]); net.eval()
        tf = T.Compose([T.ToPILImage(),T.Resize((224,224)),T.ToTensor(),
            T.Normalize([.485,.456,.406],[.229,.224,.225])])
        def get_vec(img):
            rgb = img[:,:,::-1].copy()
            t   = tf(rgb).unsqueeze(0)
            with torch.no_grad():
                f = net(t).squeeze().numpy().flatten()
            f /= np.linalg.norm(f)+1e-8
            return f
        print("MobileNetV2 fallback")

    class _Ext:
        def vec(self, img): return get_vec(img)

    fm = FusionMatcher(_Ext())

    # Load background if provided
    if len(sys.argv) >= 4:
        bg_img = cv2.imread(sys.argv[3])
        if bg_img is not None:
            bg_img   = cv2.resize(bg_img,(128,128),cv2.INTER_AREA)
            bg_stats = compute_hsv_stats(bg_img)
            fm.set_background(bg_stats)

    ref_vec = get_vec(ref_img)
    fm.load_ref(ref_name, ref_img, [ref_vec])

    score, bd = fm.match(test_img, ref_name)

    print(f"\n{'='*50}")
    print(f"  Ref    : {sys.argv[1]}")
    print(f"  Test   : {sys.argv[2]}")
    print(f"  Profile: {bd['keyword']}")
    print(f"{'='*50}")
    print(f"  DINOv2 : {bd['dinov2']:.4f}")
    print(f"  LBP    : {bd['lbp']:.4f}")
    print(f"  HSV    : {bd['hsv']:.4f}")
    print(f"  FUSED  : {bd['fused']:.4f}")
    print(f"{'='*50}")
    print(f"  {'MATCH' if score >= 0.52 else 'NO MATCH'}")
