import numpy as np


class MapBuilder:

    def __init__(self):

        self.points = []

        print("[SLAM] Map builder initialized")


    def add_observation(self, x, y):

        self.points.append((x,y))


    def get_map(self):

        return np.array(self.points)


    def clear(self):

        self.points = []