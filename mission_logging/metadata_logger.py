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

        self.records = []
        self.detection_counts = {}
        self.instance_counts = {}

        print(f"[LOGGER] Logging to {self.log_path}")


    # ----------------------------------------------------
    # Log each detection event
    # ----------------------------------------------------

    def log_detection(self,
                      frame_index,
                      feature_type,
                      detection,
                      image_path,
                      drone_pose=None,
                      instance_id=None,
                      is_new_instance=False):

        # Update detection counters
        self.detection_counts[feature_type] = \
            self.detection_counts.get(feature_type, 0) + 1

        if is_new_instance:

            self.instance_counts[feature_type] = \
                self.instance_counts.get(feature_type, 0) + 1

        record = {

            "timestamp": datetime.now().isoformat(),

            "frame_index": frame_index,

            "feature_type": feature_type,

            "instance_id": instance_id,

            "is_new_instance": is_new_instance,

            "confidence": detection.get("confidence", 0),

            "bbox": detection.get("bbox"),

            "bbox_center": detection.get("bbox_center"),

            "raw_detection_count": self.detection_counts[feature_type],

            "unique_instance_count": self.instance_counts.get(feature_type, 0),

            "drone_pose": drone_pose or {"x":0,"y":0,"z":0},

            "image_path": image_path
        }

        self.records.append(record)

        self._save()


    # ----------------------------------------------------
    # Save final instance summary
    # ----------------------------------------------------

    def log_instance_summary(self, tracker_summary):

        summary_path = self.log_path.replace(".json", "_instances.json")

        with open(summary_path, "w") as f:

            json.dump(tracker_summary, f, indent=2)

        print(f"[LOGGER] Instance summary saved → {summary_path}")


    # ----------------------------------------------------
    # Save detection log
    # ----------------------------------------------------

    def _save(self):

        with open(self.log_path, "w") as f:

            json.dump({
                "detection_counts": self.detection_counts,
                "instance_counts": self.instance_counts,
                "records": self.records
            }, f, indent=2)