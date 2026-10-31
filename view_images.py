# import os
# import matplotlib.pyplot as plt
# from PIL import Image

# base_path = r"C:\Ashish\FINAL YEAR PROJECT\Training"

# classes = ["glioma", "meningioma", "notumor", "pituitary"]

# plt.figure(figsize=(10, 10))

# for i, class_name in enumerate(classes):

#     class_path = os.path.join(base_path, class_name)

#     files = os.listdir(class_path)

#     image_path = os.path.join(class_path, files[0])

#     image = Image.open(image_path)

#     plt.subplot(2, 2, i + 1)
#     plt.imshow(image, cmap="gray")
#     plt.title(class_name)
#     plt.axis("off")

# plt.tight_layout()
# plt.show()

import os
import matplotlib.pyplot as plt
from PIL import Image

base_path = r"C:\Ashish\FINAL YEAR PROJECT\Training"

classes = ["glioma", "meningioma", "notumor", "pituitary"]

plt.figure(figsize=(12, 12))

for row, class_name in enumerate(classes):

    class_path = os.path.join(base_path, class_name)

    files = os.listdir(class_path)[:4]

    for col, file_name in enumerate(files):

        image_path = os.path.join(class_path, file_name)

        image = Image.open(image_path)

        position = row * 4 + col + 1

        plt.subplot(4, 4, position)
        plt.imshow(image, cmap="gray")
        plt.title(class_name)
        plt.axis("off")

plt.tight_layout()
plt.show()