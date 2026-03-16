import numpy as np


class LawnmowerPlanner:

    def __init__(self, width, height, row_spacing):

        self.width = width
        self.height = height
        self.row_spacing = row_spacing

        print("[NAV] Lawnmower planner ready")


    def generate_path(self):

        waypoints = []

        y = 0
        direction = 1

        while y <= self.height:

            if direction == 1:
                waypoints.append((0, y))
                waypoints.append((self.width, y))

            else:
                waypoints.append((self.width, y))
                waypoints.append((0, y))

            y += self.row_spacing
            direction *= -1

        print(f"[NAV] Generated {len(waypoints)} waypoints")

        return waypoints


    def compute_velocity(self, current_pos, target_pos, speed=0.5):

        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.2:
            return (0, 0)

        vx = speed * dx / distance
        vy = speed * dy / distance

        return vx, vy


    def waypoint_reached(self, current_pos, target_pos, tolerance=0.3):

        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        dist = np.sqrt(dx**2 + dy**2)

        return dist < tolerance