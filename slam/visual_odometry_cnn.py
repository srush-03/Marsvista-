import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms


class VisualOdometryCNN:

    def __init__(self):

        model = models.resnet18(weights="DEFAULT")
        model.eval()

        self.extractor = torch.nn.Sequential(*list(model.children())[:-1])

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        self.prev_vec = None

        self.x = 0
        self.y = 0

        print("[SLAM] CNN Visual Odometry ready")


    def process_frame(self, frame):

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            vec = self.extractor(tensor).flatten().numpy()

        if self.prev_vec is None:
            self.prev_vec = vec
            return self.get_pose()

        movement = np.linalg.norm(vec - self.prev_vec)

        self.x += movement * 0.1
        self.y += movement * 0.1

        self.prev_vec = vec

        return self.get_pose()


    def get_pose(self):

        return {
            "x": self.x,
            "y": self.y
        }