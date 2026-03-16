import time
import psutil


class HealthMonitor:

    def __init__(self, pixhawk_controller=None, battery_threshold=20):

        self.pixhawk = pixhawk_controller
        self.battery_threshold = battery_threshold

        self.running = True

        print("[HEALTH] Health monitor initialized")


    # -------------------------------------------------
    # Battery Status
    # -------------------------------------------------

    def get_battery_level(self):

        if self.pixhawk is not None:
            try:
                return self.pixhawk.get_battery_percentage()
            except:
                pass

        # fallback if Pixhawk unavailable
        return 100


    # -------------------------------------------------
    # CPU Temperature
    # -------------------------------------------------

    def get_cpu_temperature(self):

        try:
            temps = psutil.sensors_temperatures()

            if "cpu_thermal" in temps:
                return temps["cpu_thermal"][0].current

        except:
            pass

        return None


    # -------------------------------------------------
    # Memory Usage
    # -------------------------------------------------

    def get_memory_usage(self):

        mem = psutil.virtual_memory()

        return mem.percent


    # -------------------------------------------------
    # System Status
    # -------------------------------------------------

    def system_status(self):

        battery = self.get_battery_level()

        cpu_temp = self.get_cpu_temperature()

        memory = self.get_memory_usage()

        status = {

            "battery": battery,
            "cpu_temp": cpu_temp,
            "memory": memory
        }

        return status


    # -------------------------------------------------
    # Safety Check
    # -------------------------------------------------

    def safety_check(self):

        status = self.system_status()

        battery = status["battery"]

        if battery is not None:

            if battery < self.battery_threshold:

                print(
                    f"[HEALTH] Battery critical ({battery}%)"
                )

                return False

        return True


    # -------------------------------------------------
    # Continuous Monitoring
    # -------------------------------------------------

    def monitor_loop(self, check_interval=2):

        print("[HEALTH] Monitoring started")

        while self.running:

            status = self.system_status()

            battery = status["battery"]
            cpu = status["cpu_temp"]
            memory = status["memory"]

            print(
                f"[HEALTH] Battery={battery}% | CPU={cpu}°C | RAM={memory}%"
            )

            if not self.safety_check():

                print("[HEALTH] ABORT MISSION")

                if self.pixhawk is not None:
                    self.pixhawk.return_to_base()

                break

            time.sleep(check_interval)


    # -------------------------------------------------
    # Stop Monitor
    # -------------------------------------------------

    def stop(self):

        self.running = False

        print("[HEALTH] Monitoring stopped")