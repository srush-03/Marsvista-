from navigation.boundary_mapper import BoundaryMapper
from navigation.lawnmower_planner import LawnmowerPlanner


class MissionPlanner:

    def __init__(self):

        self.boundary_mapper = BoundaryMapper()

        self.lawnmower = None
        self.path = []
        self.current_index = 0

        print("[NAV] Mission planner initialized")


    # ------------------------------------------
    # Add boundary observation
    # ------------------------------------------

    def update_boundary(self, position, boundary_position):

        self.boundary_mapper.add_observation(
            position,
            boundary_position
        )


    # ------------------------------------------
    # Check if arena discovered
    # ------------------------------------------

    def boundary_complete(self):

        return self.boundary_mapper.boundary_complete()


    # ------------------------------------------
    # Initialize search path
    # ------------------------------------------

    def generate_search_path(self, row_spacing=1):

        size = self.boundary_mapper.arena_size()

        if size is None:

            print("[NAV] Arena size unknown")

            return False

        width, height = size

        print(f"[NAV] Arena estimated: {width:.2f} x {height:.2f}")

        self.lawnmower = LawnmowerPlanner(width, height, row_spacing)

        self.path = self.lawnmower.generate_path()

        self.current_index = 0

        return True


    # ------------------------------------------
    # Next waypoint
    # ------------------------------------------

    def get_next_waypoint(self):

        if self.current_index >= len(self.path):

            return None

        wp = self.path[self.current_index]

        return wp


    # ------------------------------------------
    # Update progress
    # ------------------------------------------

    def update_position(self, current_pos):

        if self.lawnmower is None:
            return None

        target = self.get_next_waypoint()

        if target is None:
            return None

        if self.lawnmower.waypoint_reached(current_pos, target):

            self.current_index += 1

        return target


    # ------------------------------------------
    # Mission complete
    # ------------------------------------------

    def mission_complete(self):

        return self.current_index >= len(self.path)