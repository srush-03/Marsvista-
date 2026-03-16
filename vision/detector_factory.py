from config.config_loader import ConfigLoader


class DetectorFactory:

    @staticmethod
    def create_detector():

        config = ConfigLoader().get_all()

        detector_type = config.get("detector", "orb")
        seed_path = config.get("seed_path", "seeds")

        print(f"[VISION] Initializing detector: {detector_type}")

        if detector_type == "cnn":

            from vision.cnn.cnn_seed_loader import CNNSeedLoader
            from vision.cnn.cnn_matcher import CNNMatcher

            loader = CNNSeedLoader(seed_path)
            detector = CNNMatcher(loader)

            print("[VISION] CNN detector loaded")

            return detector

        elif detector_type == "orb":

            from vision.orb.orb_seed_loader import ORBSeedLoader
            from vision.orb.orb_matcher import ORBMatcher

            loader = ORBSeedLoader(seed_path)
            detector = ORBMatcher(loader)

            print("[VISION] ORB detector loaded")

            return detector

        else:
            raise ValueError(f"Unknown detector type: {detector_type}")