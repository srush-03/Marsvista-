class PoseEstimator:

    def __init__(self):

        self.pose = {
            "x": 0,
            "y": 0,
            "z": 0
        }


    def update(self, new_pose):

        self.pose["x"] = new_pose["x"]
        self.pose["y"] = new_pose["y"]

        return self.pose


    def get_pose(self):

        return self.pose