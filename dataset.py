import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np


class BrainTumorDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        image_path = self.data.iloc[index]["image_path"]
        label = self.data.iloc[index]["label"]

        # Read as grayscale
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        # Gaussian filtering
        image = cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=1.5
        )

        # Resize
        image = cv2.resize(
            image,
            (224, 224)
        )

        # Normalize 0 → 1
        image = image.astype(np.float32) / 255.0

        # Convert:
        # 224 × 224
        #      ↓
        # 1 × 224 × 224
        image = torch.tensor(image).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label