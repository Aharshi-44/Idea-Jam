from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import models, transforms


class FeatureExtractor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def extract_embedding(self, roi_bgr: np.ndarray) -> np.ndarray:
        if roi_bgr is None or roi_bgr.size == 0:
            raise ValueError("ROI is empty or invalid.")

        if roi_bgr.ndim == 2:
            roi_bgr = cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR)

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(roi_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor).flatten(start_dim=1)

        return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)

