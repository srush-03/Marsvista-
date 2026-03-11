
import time
import math

class InstanceTracker:
    """
    Clusters detections across frames to avoid double counting.

    Problem this solves:
    - UAV hovers over rock feature for 5 seconds
    - CNN matcher fires 30 detections of same rock
    - We should count it as ONE instance, not 30

    Solution:
    - Each detection has a bbox_center (pixel x, y)
    - We also track time
    - If a new detection is close to an existing instance
      AND recent in time → it's the same instance
    - If it's far away → it's a NEW instance

    On Jetson this runs every frame, very lightweight.
    """

    def __init__(self,
                 pixel_distance_threshold=120,
                 time_window_seconds=4.0):
        """
        pixel_distance_threshold:
            If a new detection's bbox center is within this many
            pixels of a known instance, treat it as the same instance.
            120px works well at 720p resolution from ~3-5m altitude.

        time_window_seconds:
            After this many seconds of not seeing an instance,
            it is considered "gone" and a re-detection would
            create a new instance ID.
            Set to ~4s so brief occlusions don't create duplicates.
        """
        self.pixel_threshold = pixel_distance_threshold
        self.time_window     = time_window_seconds

        # Dict: instance_id → instance_record
        self.instances = {}
        self._id_counter = 0

    # ── PUBLIC API ────────────────────────────────────────────────

    def update(self, feature_type, bbox_center, confidence,
               frame_index, world_coords=None):
        """
        Call this for every detection coming out of CNNMatcher.

        Returns:
            (instance_id, is_new)
            instance_id : str  — unique ID for this instance
            is_new      : bool — True if this is a brand new instance
        """
        cx, cy = bbox_center
        now    = time.time()

        best_match_id = None
        best_distance = float('inf')

        for inst_id, inst in self.instances.items():
            if inst['feature_type'] != feature_type:
                continue

            age = now - inst['last_seen']
            if age > self.time_window:
                continue

            dist = self._pixel_distance(
                (cx, cy), inst['last_center']
            )

            if dist < self.pixel_threshold and dist < best_distance:
                best_distance = dist
                best_match_id = inst_id

        if best_match_id is not None:
            # ── UPDATE EXISTING INSTANCE ──────────────────
            inst = self.instances[best_match_id]
            inst['last_seen']   = now
            inst['last_center'] = (cx, cy)
            inst['hit_count']  += 1
            inst['last_frame']  = frame_index

            if confidence > inst['best_confidence']:
                inst['best_confidence'] = confidence

            if world_coords:
                inst['world_coords'] = world_coords

            return best_match_id, False

        else:
            # ── CREATE NEW INSTANCE ───────────────────────
            self._id_counter += 1
            new_id = f"{feature_type}_inst_{self._id_counter:03d}"

            self.instances[new_id] = {
                'instance_id'    : new_id,
                'feature_type'   : feature_type,
                'first_seen'     : now,
                'last_seen'      : now,
                'first_frame'    : frame_index,
                'last_frame'     : frame_index,
                'last_center'    : (cx, cy),
                'hit_count'      : 1,
                'best_confidence': confidence,
                'world_coords'   : world_coords or {"x": 0, "y": 0}
            }

            print(f"[INSTANCE TRACKER] NEW instance: {new_id} "
                  f"| center: ({cx},{cy}) "
                  f"| confidence: {confidence:.3f}")

            return new_id, True

    def get_confirmed_instances(self, min_hits=3):
        """
        Returns only instances seen at least min_hits times.
        Single-frame flash matches are false positives.
        Seen 3+ times across different frames = real detection.
        """
        confirmed = {}
        for inst_id, inst in self.instances.items():
            if inst['hit_count'] >= min_hits:
                confirmed[inst_id] = inst
        return confirmed

    def get_all_instances(self):
        return self.instances

    def get_summary(self):
        """Clean summary dict for logging. Only confirmed instances."""
        confirmed = self.get_confirmed_instances()
        summary = {}

        for inst_id, inst in confirmed.items():
            ftype = inst['feature_type']
            if ftype not in summary:
                summary[ftype] = []
            summary[ftype].append({
                'instance_id'    : inst_id,
                'hit_count'      : inst['hit_count'],
                'best_confidence': inst['best_confidence'],
                'world_coords'   : inst['world_coords'],
                'first_frame'    : inst['first_frame'],
                'last_frame'     : inst['last_frame']
            })

        return summary

    def count_confirmed(self):
        """How many confirmed unique instances per feature type."""
        counts = {}
        for inst in self.get_confirmed_instances().values():
            ftype = inst['feature_type']
            counts[ftype] = counts.get(ftype, 0) + 1
        return counts

    def _pixel_distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# ── QUICK TEST ────────────────────────────────────────────────────
if __name__ == "__main__":
    tracker = InstanceTracker(
        pixel_distance_threshold=120,
        time_window_seconds=4.0
    )

    print("=== Instance Tracker Test ===\n")

    print("Simulating 5 detections of same rock at ~same position...")
    for i in range(5):
        inst_id, is_new = tracker.update(
            feature_type="feature_type_1",
            bbox_center=(320 + i*5, 240 + i*2),
            confidence=0.82 + i * 0.01,
            frame_index=i * 10
        )
        print(f"  Frame {i*10}: id={inst_id} | new={is_new}")

    print("\nSimulating detection of different rock far away...")
    inst_id2, is_new2 = tracker.update(
        feature_type="feature_type_1",
        bbox_center=(900, 500),
        confidence=0.79,
        frame_index=60
    )
    print(f"  Frame 60: id={inst_id2} | new={is_new2}")

    print("\n=== CONFIRMED INSTANCES (min 3 hits) ===")
    summary = tracker.get_summary()
    for ftype, instances in summary.items():
        print(f"\n{ftype}:")
        for inst in instances:
            print(f"  {inst['instance_id']} | hits={inst['hit_count']} | best_conf={inst['best_confidence']:.3f}")

    print("\n=== COUNT PER TYPE ===")
    counts = tracker.count_confirmed()
    for ftype, count in counts.items():
        print(f"  {ftype}: {count} confirmed unique instances")