import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

image_path = r"C:\Ashish\FINAL YEAR PROJECT\Training\meningioma\Tr-aug-me_1.jpg"

# Load image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Gaussian filtering
gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)

# Resize
resized = cv2.resize(gaussian, (224, 224))

# Normalize to 0-1
normalized = resized / 255.0

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gaussian, cmap="gray")
plt.title("Gaussian σ=1.5")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(normalized, cmap="gray")
plt.title("224×224 + Normalized")
plt.axis("off")

plt.tight_layout()
plt.show()
