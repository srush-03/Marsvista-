import subprocess
import time
import psutil
import os
import signal


MISSION_SCRIPT = "pipeline/mission_controller.py"
CHECK_INTERVAL = 5
MAX_CPU = 95
MAX_MEMORY = 90


class Watchdog:

    def __init__(self):

        self.process = None

        print("[WATCHDOG] Supervisor started")


    # --------------------------------------
    # Start mission process
    # --------------------------------------

    def start_mission(self):

        print("[WATCHDOG] Starting mission controller")

        self.process = subprocess.Popen(
            ["python3", MISSION_SCRIPT]
        )


    # --------------------------------------
    # Check if mission running
    # --------------------------------------

    def is_running(self):

        if self.process is None:
            return False

        return self.process.poll() is None


    # --------------------------------------
    # Kill mission process
    # --------------------------------------

    def stop_mission(self):

        if self.process is None:
            return

        print("[WATCHDOG] Stopping mission")

        os.kill(self.process.pid, signal.SIGTERM)

        self.process = None


    # --------------------------------------
    # System health check
    # --------------------------------------

    def system_health(self):

        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent

        if cpu > MAX_CPU:
            print("[WATCHDOG] CPU overload")

            return False

        if memory > MAX_MEMORY:
            print("[WATCHDOG] Memory overload")

            return False

        return True


    # --------------------------------------
    # Main monitoring loop
    # --------------------------------------

    def run(self):

        self.start_mission()

        while True:

            time.sleep(CHECK_INTERVAL)

            if not self.is_running():

                print("[WATCHDOG] Mission crashed — restarting")

                self.start_mission()

                continue


            if not self.system_health():

                print("[WATCHDOG] System unstable — restarting mission")

                self.stop_mission()

                time.sleep(2)

                self.start_mission()


if __name__ == "__main__":

    watchdog = Watchdog()

    watchdog.run()