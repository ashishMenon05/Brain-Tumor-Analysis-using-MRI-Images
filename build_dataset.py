import os
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_PATH = r"C:\Ashish\FINAL YEAR PROJECT"

CLASS_NAMES = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}


def collect_images(split):

    rows = []

    split_path = os.path.join(BASE_PATH, split)

    for class_name, label in CLASS_NAMES.items():

        class_path = os.path.join(split_path, class_name)

        for filename in os.listdir(class_path):

            if filename.lower().endswith((".jpg", ".jpeg", ".png")):

                image_path = os.path.join(
                    class_path,
                    filename
                )

                rows.append({
                    "image_path": image_path,
                    "label": label,
                    "class": class_name
                })

    return pd.DataFrame(rows)


# -----------------------------------
# 1. Collect original training data
# -----------------------------------

training_df = collect_images("Training")

print("Original training:", len(training_df))


# -----------------------------------
# 2. Split training → train + validation
# -----------------------------------

train_df, validation_df = train_test_split(
    training_df,
    test_size=0.15,
    stratify=training_df["label"],
    random_state=42
)


# -----------------------------------
# 3. Collect untouched test data
# -----------------------------------
test_df = collect_images("Testing")
# -----------------------------------
# 4. Save CSV files
# -----------------------------------
train_df.to_csv("train.csv", index=False)
validation_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)
# -----------------------------------
# 5. Display results
# -----------------------------------
print("\nFinal dataset split:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))
print("\nTraining class distribution:")
print(train_df["class"].value_counts())
print("\nValidation class distribution:")
print(validation_df["class"].value_counts())
print("\nTesting class distribution:")
print(test_df["class"].value_counts())