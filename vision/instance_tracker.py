import time
import math


class InstanceTracker:

    def __init__(self,
                 pixel_distance_threshold=120,
                 time_window_seconds=4.0):

        self.pixel_threshold = pixel_distance_threshold
        self.time_window = time_window_seconds

        self.instances = {}
        self._id_counter = 0


    def update(self, feature_type, bbox_center,
               confidence, frame_index):

        cx, cy = bbox_center
        now = time.time()

        best_match = None
        best_dist = float("inf")

        for inst_id, inst in self.instances.items():

            if inst["feature_type"] != feature_type:
                continue

            age = now - inst["last_seen"]

            if age > self.time_window:
                continue

            dist = math.sqrt(
                (cx - inst["last_center"][0]) ** 2 +
                (cy - inst["last_center"][1]) ** 2
            )

            if dist < self.pixel_threshold and dist < best_dist:
                best_match = inst_id
                best_dist = dist


        if best_match:

            inst = self.instances[best_match]

            inst["last_seen"] = now
            inst["last_center"] = (cx, cy)
            inst["hit_count"] += 1
            inst["last_frame"] = frame_index

            if confidence > inst["best_confidence"]:
                inst["best_confidence"] = confidence

            return best_match, False

        else:

            self._id_counter += 1

            inst_id = f"{feature_type}_inst_{self._id_counter}"

            self.instances[inst_id] = {

                "feature_type": feature_type,
                "first_seen": now,
                "last_seen": now,
                "first_frame": frame_index,
                "last_frame": frame_index,
                "last_center": (cx, cy),
                "hit_count": 1,
                "best_confidence": confidence
            }

            return inst_id, True


    def get_confirmed_instances(self, min_hits=3):

        confirmed = {}

        for inst_id, inst in self.instances.items():

            if inst["hit_count"] >= min_hits:
                confirmed[inst_id] = inst

        return confirmed


    def count_confirmed(self):

        counts = {}

        for inst in self.get_confirmed_instances().values():

            ftype = inst["feature_type"]

            counts[ftype] = counts.get(ftype, 0) + 1

        return counts


    def get_summary(self):

        summary = {}

        for inst_id, inst in self.get_confirmed_instances().items():

            ftype = inst["feature_type"]

            if ftype not in summary:
                summary[ftype] = []

            summary[ftype].append({

                "instance_id": inst_id,
                "hit_count": inst["hit_count"],
                "best_confidence": inst["best_confidence"],
                "first_frame": inst["first_frame"],
                "last_frame": inst["last_frame"]
            })

        return summary