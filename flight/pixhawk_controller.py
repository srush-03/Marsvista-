import time
from dronekit import connect, VehicleMode
from pymavlink import mavutil


class PixhawkController:

    def __init__(self,
                 connection_string="/dev/ttyAMA0",
                 baud=57600):

        print("[PIXHAWK] Connecting to flight controller...")

        self.vehicle = connect(
            connection_string,
            baud=baud,
            wait_ready=True
        )

        print("[PIXHAWK] Connected")


    # ----------------------------------------
    # Vehicle Status
    # ----------------------------------------

    def get_battery_percentage(self):

        battery = self.vehicle.battery

        if battery is None:
            return None

        return battery.level


    def get_voltage(self):

        battery = self.vehicle.battery

        if battery is None:
            return None

        return battery.voltage


    def is_armable(self):

        return self.vehicle.is_armable


    # ----------------------------------------
    # Arm / Disarm
    # ----------------------------------------

    def arm(self):

        print("[PIXHAWK] Arming motors")

        while not self.vehicle.is_armable:
            print("[PIXHAWK] Waiting for vehicle to initialise...")
            time.sleep(1)

        self.vehicle.mode = VehicleMode("GUIDED")

        self.vehicle.armed = True

        while not self.vehicle.armed:
            print("[PIXHAWK] Waiting for arming...")
            time.sleep(1)

        print("[PIXHAWK] Vehicle armed")


    def disarm(self):

        print("[PIXHAWK] Disarming")

        self.vehicle.armed = False

        while self.vehicle.armed:
            time.sleep(1)

        print("[PIXHAWK] Disarmed")


    # ----------------------------------------
    # Takeoff
    # ----------------------------------------

    def takeoff(self, target_altitude=4.5):

        print(f"[PIXHAWK] Taking off to {target_altitude} meters")

        self.vehicle.simple_takeoff(target_altitude)

        while True:

            alt = self.vehicle.location.global_relative_frame.alt

            print(f"[PIXHAWK] Altitude: {alt:.2f}")

            if alt >= target_altitude * 0.95:
                print("[PIXHAWK] Target altitude reached")
                break

            time.sleep(1)


    # ----------------------------------------
    # Velocity Control
    # ----------------------------------------

    def send_velocity(self, vx, vy, vz, duration=1):

        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            0,
            0,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,
            0,
            0,
            0,
            vx,
            vy,
            vz,
            0,
            0,
            0,
            0,
            0
        )

        for _ in range(duration):

            self.vehicle.send_mavlink(msg)

            time.sleep(1)


    # ----------------------------------------
    # Return to Base
    # ----------------------------------------

    def return_to_base(self):

        print("[PIXHAWK] Returning to base")

        self.vehicle.mode = VehicleMode("RTL")

        while self.vehicle.mode.name != "RTL":
            time.sleep(1)


    # ----------------------------------------
    # Land
    # ----------------------------------------

    def land(self):

        print("[PIXHAWK] Landing")

        self.vehicle.mode = VehicleMode("LAND")

        while self.vehicle.armed:
            time.sleep(1)

        print("[PIXHAWK] Land complete")


    # ----------------------------------------
    # Shutdown
    # ----------------------------------------

    def close(self):

        print("[PIXHAWK] Closing connection")

        self.vehicle.close()