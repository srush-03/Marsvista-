import cv2
import numpy as np
import os


class ORBSeedLoader:

    def __init__(self, seeds_folder):

        print("[ORB] Initializing ORB seed loader")

        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8
        )

        self.seed_descriptors = {}

        self._load_all(seeds_folder)


    def _extract_features(self, image_path):

        img = cv2.imread(image_path)

        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        return descriptors


    def _load_all(self, folder):

        for feature_type in os.listdir(folder):

            type_path = os.path.join(folder, feature_type)

            if not os.path.isdir(type_path):
                continue

            print(f"[ORB] Loading {feature_type}")

            all_desc = []

            for img_file in os.listdir(type_path):

                if not img_file.lower().endswith(
                    ('.jpg', '.jpeg', '.png')
                ):
                    continue

                img_path = os.path.join(type_path, img_file)

                desc = self._extract_features(img_path)

                if desc is not None:
                    all_desc.append(desc)

            if all_desc:
                self.seed_descriptors[feature_type] = np.vstack(all_desc)

        print("[ORB] Seed loading complete")


    def get_descriptors(self):

        return self.seed_descriptors