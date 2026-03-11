import json
import os
from datetime import datetime


class MetadataLogger:
    def __init__(self, log_folder):
        os.makedirs(log_folder, exist_ok=True)
        self.log_path = os.path.join(
            log_folder,
            f"flight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        self.records          = []
        self.detection_counts = {}   # raw hit counts per type
        self.instance_counts  = {}   # unique instance counts per type

    def log_detection(self, frame_index, feature_type, detection,
                      image_path, image_id=None,
                      drone_pose=None, instance_id=None,
                      is_new_instance=False):

        # Raw detection counter
        self.detection_counts[feature_type] = \
            self.detection_counts.get(feature_type, 0) + 1

        # Unique instance counter
        if is_new_instance:
            self.instance_counts[feature_type] = \
                self.instance_counts.get(feature_type, 0) + 1

        record = {
            # ── IDENTIFICATION ──────────────────────────────
            "detection_id"          : f"det_{frame_index:06d}_{feature_type}",
            "image_id"              : image_id or os.path.basename(image_path),
            "timestamp"             : datetime.now().isoformat(),
            "frame_index"           : frame_index,

            # ── FEATURE INFO ────────────────────────────────
            "feature_type"          : feature_type,
            "instance_id"           : instance_id or "untracked",
            "is_new_instance"       : is_new_instance,

            # ── COUNTS ──────────────────────────────────────
            "raw_detection_count"   : self.detection_counts[feature_type],
            "unique_instance_count" : self.instance_counts.get(feature_type, 0),

            # ── MATCH QUALITY ───────────────────────────────
            "confidence"            : detection.get('confidence', 0.0),
            "inliers"               : detection.get('inliers', -1),   # SIFT only
            "votes"                 : detection.get('votes', -1),     # SIFT only

            # ── GEOMETRY ────────────────────────────────────
            "bbox"                  : self._serialize_bbox(detection.get('bbox')),
            "bbox_center"           : list(detection.get('bbox_center', [0, 0])),

            # ── POSITION ────────────────────────────────────
            "drone_pose"            : drone_pose or {"x": 0, "y": 0, "z": 4.5},
            "world_coordinates"     : {"x": 0, "y": 0},   # filled by VIO later

            # ── FILE PATH ───────────────────────────────────
            "verification_image_path": image_path,
        }

        self.records.append(record)
        self._save()

        flag = " ★ NEW" if is_new_instance else ""
        print(f"[LOG] {feature_type} | {instance_id} "
              f"| conf={detection.get('confidence', 0):.3f}"
              f" | img={image_id}{flag}")

        return record

    def log_instance_summary(self, tracker_summary):
        """
        Saves final per-instance summary for base station validation.
        Called once at end of flight.
        """
        summary_path = self.log_path.replace(".json", "_instances.json")
        with open(summary_path, 'w') as f:
            json.dump(tracker_summary, f, indent=2)
        print(f"[LOG] Instance summary → {summary_path}")

    def _serialize_bbox(self, bbox):
        if bbox is None:
            return None
        if isinstance(bbox, (tuple, list)):
            return list(bbox)
        try:
            import numpy as np
            if isinstance(bbox, np.ndarray):
                return bbox.tolist()
        except Exception:
            pass
        return str(bbox)

    def _save(self):
        with open(self.log_path, 'w') as f:
            json.dump({
                "detection_counts" : self.detection_counts,
                "instance_counts"  : self.instance_counts,
                "records"          : self.records
            }, f, indent=2)