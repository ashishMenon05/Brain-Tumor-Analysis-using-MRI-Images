# import os

# base_path = r"C:\Ashish\FINAL YEAR PROJECT"

# for split in ["Training", "Testing"]:

#     split_path = os.path.join(base_path, split)

#     print(f"\n===== {split} =====")

#     total = 0

#     for class_name in os.listdir(split_path):

#         class_path = os.path.join(split_path, class_name)

#         if os.path.isdir(class_path):

#             images = os.listdir(class_path)

#             count = len(images)
#             total += count

#             print(f"{class_name}: {count}")

#     print(f"Total: {total}")

import os
from PIL import Image

base_path = r"C:\Ashish\FINAL YEAR PROJECT"

for split in ["Training", "Testing"]:

    split_path = os.path.join(base_path, split)

    print(f"\n===== {split} =====")

    for class_name in os.listdir(split_path):

        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            continue

        files = os.listdir(class_path)

        # Take the first image
        first_image = files[0]
        image_path = os.path.join(class_path, first_image)

        image = Image.open(image_path)

        print(
            f"{class_name}: "
            f"example={first_image}, "
            f"size={image.size}, "
            f"mode={image.mode}, "
            f"format={image.format}"
        )