from config.config_loader import ConfigLoader


class SLAMFactory:

    @staticmethod
    def create_slam():

        config = ConfigLoader().get_all()

        slam_type = config.get("slam", "orb")

        print(f"[SLAM] Initializing SLAM: {slam_type}")

        if slam_type == "cnn":
            from slam.visual_odometry_cnn import VisualOdometryCNN
            return VisualOdometryCNN()

        elif slam_type == "orb":
            from slam.visual_odometry_orb import VisualOdometryORB
            return VisualOdometryORB()

        else:
            raise ValueError("Unknown SLAM type")