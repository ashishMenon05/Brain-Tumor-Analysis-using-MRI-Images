import cv2
import numpy as np


def preprocess_image(image_path):

    # 1. Read image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. Gaussian filtering
    image = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=1.5
    )

    # 3. Resize
    image = cv2.resize(
        image,
        (224, 224)
    )

    # 4. Normalize pixel values
    image = image.astype(np.float32) / 255.0

    # 5. Add channel dimension
    image = np.expand_dims(image, axis=-1)

    return image