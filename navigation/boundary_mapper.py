import numpy as np


class BoundaryMapper:
    """
    Builds an estimated arena boundary from yellow line detections.

    The drone records the pose whenever it detects a boundary line.
    Over time this allows estimation of:

        xmin, xmax
        ymin, ymax

    which define the exploration region.
    """

    def __init__(self):

        self.left_points = []
        self.right_points = []
        self.top_points = []
        self.bottom_points = []

        print("[BOUNDARY] Mapper initialized")


    # ---------------------------------------------------
    # Add new boundary observation
    # ---------------------------------------------------

    def add_observation(self, position, boundary_position):

        x, y = position

        if boundary_position == "left":
            self.left_points.append((x, y))

        elif boundary_position == "right":
            self.right_points.append((x, y))

        elif boundary_position == "top":
            self.top_points.append((x, y))

        elif boundary_position == "bottom":
            self.bottom_points.append((x, y))


    # ---------------------------------------------------
    # Estimate arena limits
    # ---------------------------------------------------

    def estimate_bounds(self):

        xmin = None
        xmax = None
        ymin = None
        ymax = None

        if self.left_points:
            xmin = np.mean([p[0] for p in self.left_points])

        if self.right_points:
            xmax = np.mean([p[0] for p in self.right_points])

        if self.bottom_points:
            ymin = np.mean([p[1] for p in self.bottom_points])

        if self.top_points:
            ymax = np.mean([p[1] for p in self.top_points])

        return xmin, xmax, ymin, ymax


    # ---------------------------------------------------
    # Check if boundary is fully discovered
    # ---------------------------------------------------

    def boundary_complete(self):

        return (
            len(self.left_points) > 5 and
            len(self.right_points) > 5 and
            len(self.top_points) > 5 and
            len(self.bottom_points) > 5
        )


    # ---------------------------------------------------
    # Generate arena dimensions
    # ---------------------------------------------------

    def arena_size(self):

        xmin, xmax, ymin, ymax = self.estimate_bounds()

        if None in (xmin, xmax, ymin, ymax):
            return None

        width = xmax - xmin
        height = ymax - ymin

        return width, height