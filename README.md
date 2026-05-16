# ASCEND Perception System — Team Marsvista
## ISRO IRoC-U 2026 | Elimination Round

---

## Folder Structure

```
ascend_system/
    reference_creator.py   ← Step 1: create LR references before flight
    frame_recorder.py      ← Step 2: capture HD frames during flight
    post_match.py          ← Step 3: match frames after landing
    validate.py            ← Step 4: visual validation + resortie decision
    requirements.txt
    refs/                  ← LR reference images go here
    flight_frames/         ← HD frames saved here during flight
    results/               ← Matching results, proof images, report
```

---

## Install

```bash
pip install -r requirements.txt
```

DINOv2 is loaded automatically via torch.hub on first run (requires internet).
On Jetson without internet: set `--no-dinov2` flag to use MobileNetV2 fallback.

---

## Step-by-Step Usage

### Step 1 — Create LR References (before flight)

**Elimination round** (you create the references):
```bash
python reference_creator.py
```
- Point camera top-down at each feature from ~1.5m height
- Press **C** to capture, **S** to save, **N** for next feature
- Saves `refs/ref_0_rock_LR.jpg`, `refs/ref_1_oxide_LR.jpg`, `refs/ref_2_ice_LR.jpg`

**Final round** (ISRO provides reference images):
```bash
# Copy ISRO's images to refs/ then validate:
python reference_creator.py --validate
```

**Import from existing photo:**
```bash
python reference_creator.py --source my_photo.jpg --name ref_0_rock
```

---

### Step 2 — Record HD Frames During Flight

**Laptop/webcam testing:**
```bash
python frame_recorder.py --mode webcam
```
- Press **SPACE** to start/stop recording
- Arrow keys simulate drone movement (for testing)
- Saves frames to `flight_frames/`

**Jetson + ROS2 deployment:**
```bash
source /opt/ros/humble/setup.bash
python frame_recorder.py --mode ros2
```
Subscribes to:
- `/orb_slam3/pose` — ORB-SLAM3 camera pose
- `/oak/rgb/image_rect` — OAK-D Lite RGB frames

---

### Step 3 — Post-Flight Matching (after landing)

```bash
python post_match.py
```

If no matches found, try lowering threshold:
```bash
python post_match.py --threshold 0.45
```

Outputs:
- `results/results.json` — coordinates + proof image paths
- `results/proof_*.jpg` — HD proof images with detection boxes
- `results/compare_*.jpg` — side-by-side LR reference vs live comparison

---

### Step 4 — Validate Results

```bash
python validate.py --open
```

Opens `results/validation_report.html` in browser showing:
- All matched features with confidence scores
- Coordinates relative to base station
- HD proof images
- Re-sortie recommendation if any feature is missing/low confidence

---

## Algorithm Summary

```
reference_creator.py:
    HD photo → INTER_AREA downsample → CLAHE enhance → 128x128 LR ref

frame_recorder.py:
    ROS2 pose topic → x,y,z per frame
    OAK-D Lite RGB → 1280x720 HD frame
    Filename: frame_000001_x2.340_y1.560_z4.500.jpg

post_match.py:
    For each HD frame:
        Sliding window: 256x256 windows, stride=128 across 1280x720
        Each window → 128x128 LR
        Stage 1: SSIM pre-filter vs all 3 refs (~2ms, rejects 90%)
        Stage 2: DINOv2 ViT-S/14 cosine similarity (~25ms on candidates)
        Cluster matches by 1m coordinate proximity
        Keep top-3 per feature by CNN score

validate.py:
    Load results.json → HTML report with images
    Confidence: HIGH ≥0.75, MEDIUM ≥0.60, LOW ≥0.45
    Trigger re-sortie if any feature missing or LOW confidence
```

---

## Why DINOv2 over LightGlue

LightGlue requires keypoints. Geological surface patches (oxide powder,
rock formations, ice reflections) have almost no keypoints — they are
texture problems. At 128x128 resolution there are ~10 keypoints maximum.

DINOv2 was trained with self-supervised learning on diverse imagery and
learns texture representations directly. It produces 384-d feature vectors
that are significantly more discriminative than MobileNetV2 (1280-d,
ImageNet object classifier) for surface texture matching.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| No matches found | Threshold too high | Run with `--threshold 0.45` |
| Too many false matches | Threshold too low | Run with `--threshold 0.65` |
| DINOv2 download fails | No internet on Jetson | System auto-falls back to MobileNetV2 |
| Camera index error | Wrong camera | Script tries indices 0,1,2 automatically |
| Blurry frames | Motion / focus | Frames still saved; post_match deprioritises them |
| SLAM pose missing | SLAM3 not running | Frames saved with x0 y0 z0; matching still works |

---

## Output Format (results.json)

```json
{
  "team": "Team Marsvista",
  "challenge": "ISRO IRoC-U 2026 ASCEND",
  "features_matched": 3,
  "matches": [
    {
      "reference": "ref_0_rock_LR.jpg",
      "rank": 0,
      "coordinates": {"x": 2.34, "y": 1.56, "z": 4.50},
      "ssim_score": 0.312,
      "cnn_score": 0.718,
      "hd_proof_image": "proof_ref_0_rock_LR_rank0_fr000123.jpg",
      "lr_comparison": "compare_ref_0_rock_LR_rank0_fr000123.jpg"
    }
  ]
}
```
