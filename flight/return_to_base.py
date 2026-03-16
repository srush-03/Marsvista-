import time


class ReturnToBase:

    def __init__(self, pixhawk_controller, slam, tolerance=0.5):

        self.pixhawk = pixhawk_controller
        self.slam = slam
        self.tolerance = tolerance

        print("[RTB] Return-to-base module ready")


    # -------------------------------------------------
    # Check if drone reached base
    # -------------------------------------------------

    def at_base(self):

        pose = self.slam.get_pose()

        x = pose["x"]
        y = pose["y"]

        distance = (x**2 + y**2) ** 0.5

        return distance < self.tolerance


    # -------------------------------------------------
    # Navigate toward base
    # -------------------------------------------------

    def navigate_to_base(self, speed=0.5):

        pose = self.slam.get_pose()

        x = pose["x"]
        y = pose["y"]

        dx = -x
        dy = -y

        distance = (dx**2 + dy**2) ** 0.5

        if distance < self.tolerance:
            return True

        vx = speed * dx / distance
        vy = speed * dy / distance

        self.pixhawk.send_velocity(vx, vy, 0, duration=1)

        return False


    # -------------------------------------------------
    # Execute return procedure
    # -------------------------------------------------

    def execute(self):

        print("[RTB] Returning to base...")

        while True:

            if self.at_base():

                print("[RTB] Base reached")

                break

            reached = self.navigate_to_base()

            if reached:
                break

            time.sleep(1)

        print("[RTB] Landing")

        self.pixhawk.land()