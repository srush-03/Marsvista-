# Autonomous Drone Perception & Mission System

## Overview

This project implements a modular **autonomous drone perception and mission control system** designed for **GPS-denied environments** such as Mars-like terrains.
The system integrates computer vision, SLAM, mission planning, and flight control to allow a drone to:

* Take off from a base station
* Search an arena for **reference objects (seeds)**
* Detect and log those objects with coordinates
* Avoid boundary lines
* Map the terrain using keyframes
* Return safely to the base station
* Archive mission results

The architecture supports both **ORB (CPU-efficient)** and **CNN (GPU-accelerated)** detection pipelines so the system can run on:

* **Raspberry Pi 5 (ORB)**
* **NVIDIA Jetson (CNN)**

---

# System Architecture

```
Camera
   ↓
Async Pipeline
   ↓
Object Detector (ORB / CNN)
   ↓
Instance Tracker
   ↓
Yellow Boundary Detection
   ↓
SLAM / Visual Odometry
   ↓
Mission Planner
   ↓
Pixhawk Flight Controller
   ↓
Metadata Logger
```

The perception system runs using a **multi-threaded pipeline** to maximize FPS and reduce processing latency.

---

# Key Features

### Object Detection

Supports two interchangeable detection backends:

| Mode | Platform     | Description               |
| ---- | ------------ | ------------------------- |
| ORB  | Raspberry Pi | Fast CPU feature matching |
| CNN  | Jetson       | Deep feature similarity   |

Reference images are stored as **seed datasets**.

---

### Instance Tracking

Prevents duplicate detections of the same object by clustering detections across frames.

---

### Boundary Detection

Detects **yellow arena boundary lines** and provides navigation warnings.

---

### SLAM / Visual Odometry

Tracks drone movement without GPS by estimating position from camera motion.

---

### Mission Planning

Supports coverage search using a **lawnmower path** across the arena.

---

### Mission Logging

All detections and mission events are logged to structured JSON files.

---

# Project Structure

```
project/
│
camera/
   camera_manager.py
│
vision/
   detector_factory.py
   instance_tracker.py
   yellow_line_detector.py
   orb/
   cnn/
│
slam/
   visual_odometry_orb.py
   visual_odometry_cnn.py
   pose_estimator.py
   map_builder.py
│
navigation/
   boundary_mapper.py
   lawnmower_planner.py
   mission_planner.py
│
flight/
   pixhawk_controller.py
   health_monitor.py
   return_to_base.py
│
logging/
   metadata_logger.py
│
pipeline/
   mission_controller.py
│
config/
   config_loader.py
   mission_config.yaml
│
tests/
│
scripts/
│
data/
   archive/
   detections/
   keyframes/
   logs/
│
seeds/
│
run_missions.sh
requirements.txt
```

---

# Installation

## 1. Clone Repository

```
git clone <repository_url>
cd drone_project
```

---

## 2. Create Virtual Environment

```
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```
pip install -r requirements.txt
```

Example dependencies:

```
opencv-python
numpy
pyyaml
torch
torchvision
pymavlink
psutil
depthai
```

---

# Configuration

Configuration is controlled via:

```
config/mission_config.yaml
```

Example:

```
vision: orb
slam: orb

frame_width: 1280
frame_height: 720
camera_fps: 30

target_count: 3

planner:
  row_spacing: 1.0

battery:
  abort_threshold: 20
```

Switch detection mode easily:

```
vision: orb
```

or

```
vision: cnn
```

---

# Seed Dataset

Objects the drone must detect are stored in:

```
seeds/
```

Example structure:

```
seeds/
   rock_sample/
       img1.jpg
       img2.jpg
```

Each folder represents a **feature type**.

---

# Running Tests

Before flying the drone, verify all modules:

```
python tests/test_camera.py
python tests/test_calibration.py
python tests/test_matcher.py
python tests/test_fps.py
python tests/test_detector.py
python tests/test_boundary_detection.py
python tests/test_navigation.py
```

---

# Running the System

## Start the Mission Controller

```
python pipeline/mission_controller.py
```

The system will:

1. Start camera feed
2. Run detection pipeline
3. Track detected objects
4. Detect arena boundaries
5. Log detections
6. Complete mission when all targets are found

---

## Run Continuous Missions

```
chmod +x run_missions.sh
./run_missions.sh
```

The script will:

```
load seed folder
run mission
return to base
archive results
start next mission
```

---

# Output Data

After each mission, results are saved in:

```
data/
```

### Detections

```
data/detections/
feature_type/
det_feature_inst_frame.jpg
```

---

### Keyframes

```
data/keyframes/
kf_000120.jpg
```

Used for terrain mapping.

---

### Logs

```
data/logs/
flight_timestamp.json
flight_timestamp_instances.json
```

Contains detection metadata and mission summaries.

---

### Archived Missions

```
data/archive/
timestamp/
```

Stores completed missions for analysis.

---

# Mission Completion

A mission completes when:

```
all reference objects are detected
```

The system then:

```
creates mission_complete.flag
returns to base
archives mission data
```

---

# Hardware

Recommended hardware setup:

| Component      | Description           |
| -------------- | --------------------- |
| Raspberry Pi 5 | main onboard computer |
| OAK-D Lite     | AI stereo camera      |
| Pixhawk        | flight controller     |
| LiPo Battery   | drone power           |

Future upgrade:

```
NVIDIA Jetson → CNN detection
```

---

# Safety Notes

Before real flight always verify:

```
camera functionality
detection accuracy
boundary detection
planner path
battery monitoring
```

Never deploy without running test modules.

---

# Future Improvements

Potential upgrades include:

* Full SLAM mapping
* GPU acceleration
* multi-camera perception
* adaptive coverage planning
* terrain stitching
* real-time telemetry dashboard

---

# License

This project is intended for **research and educational robotics development**.

---

# Author

Autonomous drone perception system developed for **GPS-denied exploration and search missions**.
