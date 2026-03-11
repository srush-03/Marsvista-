import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class CNNSeedLoader:
    def __init__(self, seeds_folder):
        print("Loading MobileNetV2 pretrained model...")
        
        model = models.mobilenet_v2(weights='DEFAULT')
        model.eval()
        
        self.feature_extractor = torch.nn.Sequential(
         *list(model.children())[:-1],
          torch.nn.AdaptiveAvgPool2d((1, 1))  # fixes vector to 1280
        )
        self.feature_extractor.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.seed_vectors = {}
        self._load_all(seeds_folder)

    def extract_vector(self, image):
        if isinstance(image, str):
         img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
         img = Image.fromarray(
            image[:, :, ::-1]
         ).convert('RGB')
        else:
         img = image.convert('RGB')

    # Center crop to focus on object, ignore background
        w, h = img.size
        crop_size = min(w, h)
        left   = (w - crop_size) // 2
        top    = (h - crop_size) // 2
        right  = left + crop_size
        bottom = top + crop_size
        img = img.crop((left, top, right, bottom))

        tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
         features = self.feature_extractor(tensor)

        vector = features.squeeze().numpy().flatten()
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        return vector

    def _load_all(self, folder):
        for feature_type in os.listdir(folder):
            type_path = os.path.join(folder, feature_type)
            if not os.path.isdir(type_path):
                continue
            
            print(f"\nLoading feature type: {feature_type}")
            vectors = []
            
            for img_file in os.listdir(type_path):
                if not img_file.lower().endswith(
                    ('.jpg', '.jpeg', '.png')
                ):
                    continue
                img_path = os.path.join(type_path, img_file)
                vec = self.extract_vector(img_path)
                vectors.append(vec)
                print(f"  {img_file} → vector extracted OK")
            
            if vectors:
                avg_vector = np.mean(vectors, axis=0)
                avg_vector = avg_vector / (
                    np.linalg.norm(avg_vector) + 1e-8
                )
                self.seed_vectors[feature_type] = avg_vector
                print(f"  Prototype created for {feature_type} OK")

    def get_vectors(self):
        return self.seed_vectors


if __name__ == "__main__":
    loader = CNNSeedLoader("seeds")
    vectors = loader.get_vectors()
    
    print("\n===== CNN SEED LOADING RESULTS =====")
    for ftype, vec in vectors.items():
        print(f"{ftype}: vector shape = {vec.shape}")
    print(f"\nTotal feature types: {len(vectors)}")
    print("SUCCESS" if vectors else "FAILED")
