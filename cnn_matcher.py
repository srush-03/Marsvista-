import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import threading
import time


class CNNMatcher:
    """
    Calibrated CNN Matcher.

    Calibration results from your test (08-Mar-2026):
    ─────────────────────────────────────────────────
    Situation          ft1     ft2
    Close up          0.745   0.299
    1m away           0.637   0.286
    2m away           0.434   0.265
    Background        0.248   0.216
    ─────────────────────────────────────────────────
    Threshold set to 0.62:
      - Catches ft1 at close range and 1m ✅
      - Rejects background (0.248) ✅
      - ft2 seeds need to be replaced (max 0.334 is too low)
    """

    def __init__(self, cnn_loader, threshold=0.62):
        self.loader       = cnn_loader
        self.seed_vectors = cnn_loader.get_vectors()
        self.threshold    = threshold

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self._consecutive_hits = {ft: 0 for ft in self.seed_vectors}
        self._last_det         = {ft: None for ft in self.seed_vectors}
        self.CONFIRM_THRESHOLD = 3

        # Async thread setup
        self._latest_frame  = None
        self._latest_scores = {ft: 0.0 for ft in self.seed_vectors}
        self._latest_dets   = []
        self._lock          = threading.Lock()
        self._running       = True

        self._thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self._thread.start()

        print(f"CNN Matcher ready | threshold={threshold}")
        print(f"Watching: {list(self.seed_vectors.keys())}")
        print(f"Confirm threshold: {self.CONFIRM_THRESHOLD} consecutive hits")
        print(f"Async detection thread started.")

    # ── PUBLIC API ────────────────────────────────────────────────

    def submit_frame(self, frame):
        with self._lock:
            self._latest_frame = frame.copy()

    def get_all_scores(self, frame=None):
        with self._lock:
            return dict(self._latest_scores)

    def match_frame(self, frame=None):
        with self._lock:
            return list(self._latest_dets)

    def get_consecutive_hits(self):
        with self._lock:
            return dict(self._consecutive_hits)

    def stop(self):
        self._running = False

    # ── ASYNC DETECTION LOOP ──────────────────────────────────────

    def _detection_loop(self):
        while self._running:
            with self._lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.01)
                continue
            scores, dets, hits = self._process_frame(frame)
            with self._lock:
                self._latest_scores    = scores
                self._latest_dets      = dets
                self._consecutive_hits = hits

    def _process_frame(self, frame):
        crops     = self._get_multiscale_crops(frame)
        scores    = {ft: 0.0 for ft in self.seed_vectors}
        hit_frame = {ft: False for ft in self.seed_vectors}
        best_det  = {}

        for crop_info in crops:
            crop = crop_info['image']
            if crop.size == 0:
                continue
            x, y, cw, ch = crop_info['bbox']
            vec = self._extract_vector(crop)

            for feature_type, seed_vector in self.seed_vectors.items():
                similarity = float(np.dot(vec, seed_vector))

                if similarity > scores[feature_type]:
                    scores[feature_type] = round(similarity, 3)

                if similarity >= self.threshold:
                    hit_frame[feature_type] = True
                    if feature_type not in best_det or \
                       similarity > best_det[feature_type]['confidence']:
                        best_det[feature_type] = {
                            'feature_type': feature_type,
                            'confidence'  : round(similarity, 3),
                            'bbox'        : (x, y, cw, ch),
                            'bbox_center' : (x + cw//2, y + ch//2)
                        }

        # Update consecutive hit counters
        hits = dict(self._consecutive_hits)
        confirmed_dets = []

        for feature_type in self.seed_vectors:
            if hit_frame[feature_type]:
                hits[feature_type] += 1
                self._last_det[feature_type] = best_det[feature_type]
            else:
                hits[feature_type] = 0

            if hits[feature_type] >= self.CONFIRM_THRESHOLD:
                det = self._last_det.get(feature_type)
                if det:
                    confirmed_dets.append(det)

        return scores, confirmed_dets, hits

    def _get_multiscale_crops(self, frame):
        """
        18 crops across 4 scales.
        Scale 3 (3x3 grid) is key for small objects at 4.5m altitude.
        """
        h, w = frame.shape[:2]
        crops = []

        # Scale 1: full frame
        crops.append({'image': frame.copy(), 'bbox': (0, 0, w, h)})

        # Scale 2: 4 quadrants
        hw, hh = w // 2, h // 2
        for (x, y, cw, ch) in [
            (0, 0, hw, hh), (hw, 0, hw, hh),
            (0, hh, hw, hh), (hw, hh, hw, hh)
        ]:
            crops.append({'image': frame[y:y+ch, x:x+cw], 'bbox': (x, y, cw, ch)})

        # Scale 3: 3x3 grid — catches small objects from 4.5m
        tw, th = w // 3, h // 3
        for row in range(3):
            for col in range(3):
                x, y = col * tw, row * th
                crops.append({
                    'image': frame[y:y+th, x:x+tw],
                    'bbox' : (x, y, tw, th)
                })

        # Scale 4: center zoom — when drone hovers directly above feature
        cx, cy = w // 2, h // 2
        for frac in [0.5, 0.35, 0.25]:
            cw, ch = int(w * frac), int(h * frac)
            x, y   = cx - cw // 2, cy - ch // 2
            crop   = frame[max(0,y):min(h,y+ch), max(0,x):min(w,x+cw)]
            if crop.size > 0:
                crops.append({'image': crop, 'bbox': (x, y, cw, ch)})

        # Filter bad exposure
        valid = []
        for c in crops:
            if c['image'].size == 0:
                continue
            gray = cv2.cvtColor(c['image'], cv2.COLOR_BGR2GRAY)
            if 15 < gray.mean() < 240:
                valid.append(c)
        return valid

    def _extract_vector(self, image):
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image[:, :, ::-1]).convert('RGB')
        else:
            img = image.convert('RGB')

        iw, ih    = img.size
        crop_size = min(iw, ih)
        img = img.crop((
            (iw - crop_size) // 2, (ih - crop_size) // 2,
            (iw + crop_size) // 2, (ih + crop_size) // 2
        ))

        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            features = self.loader.feature_extractor(tensor)

        vector = features.squeeze().numpy().flatten()
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        return vector