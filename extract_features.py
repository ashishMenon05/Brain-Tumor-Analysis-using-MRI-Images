import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BrainTumorDataset

import numpy as np


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("=" * 60)


# ============================================================
# 2. CONFIGURATION
# ============================================================

BATCH_SIZE = 32

MODEL_PATH = "best_cnn_model.pth"

TRAIN_CSV = "train.csv"
VAL_CSV = "validation.csv"
TEST_CSV = "test.csv"


# ============================================================
# 3. CNN MODEL
# ============================================================

class BrainTumorCNN(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        # ----------------------------------------------------
        # Block 1
        # ----------------------------------------------------

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2
            )
        )

        # ----------------------------------------------------
        # Block 2
        # ----------------------------------------------------

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(
                kernel_size=2
            )
        )

        # ----------------------------------------------------
        # Block 3
        # ----------------------------------------------------

        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2
            )
        )

        # ----------------------------------------------------
        # Block 4
        # ----------------------------------------------------

        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=1024,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(
                kernel_size=2
            )
        )

        # ----------------------------------------------------
        # Feature representation
        # ----------------------------------------------------

        self.flatten = nn.Flatten()

        self.feature_layer = nn.Sequential(
            nn.Linear(
                1024 * 14 * 14,
                1024
            ),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            p=0.50
        )

        # ----------------------------------------------------
        # Final CNN classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            1024,
            num_classes
        )


    # ========================================================
    # NORMAL FORWARD PASS
    # ========================================================

    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.flatten(x)

        x = self.feature_layer(x)

        x = self.dropout(x)

        x = self.classifier(x)

        return x


    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def extract_features(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.flatten(x)

        # 1024-dimensional CNN representation

        x = self.feature_layer(x)

        return x


# ============================================================
# 4. LOAD BEST CNN
# ============================================================

model = BrainTumorCNN(
    num_classes=4
).to(device)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


print("\nBest CNN checkpoint loaded.")

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    f"Validation accuracy: "
    f"{checkpoint['val_accuracy']:.2f}%"
)


# ============================================================
# 5. LOAD DATASETS
# ============================================================

train_dataset = BrainTumorDataset(
    TRAIN_CSV
)

val_dataset = BrainTumorDataset(
    VAL_CSV
)

test_dataset = BrainTumorDataset(
    TEST_CSV
)


print("\nDataset sizes:")

print(
    "Training:",
    len(train_dataset)
)

print(
    "Validation:",
    len(val_dataset)
)

print(
    "Testing:",
    len(test_dataset)
)


# ============================================================
# 6. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 7. FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_dataset_features(
    loader,
    dataset_name
):

    print(
        f"\nExtracting features from "
        f"{dataset_name}..."
    )

    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(loader):

            images = images.to(device)

            # -----------------------------------------------
            # Extract 1024-dimensional CNN representation
            # -----------------------------------------------

            features = model.extract_features(
                images
            )

            # Move features to CPU

            features = features.cpu().numpy()

            labels = labels.numpy()

            all_features.append(
                features
            )

            all_labels.append(
                labels
            )

            # Progress

            if (
                batch_index + 1
            ) % 20 == 0:

                print(
                    f"Processed "
                    f"{batch_index + 1} batches"
                )


    # Combine all batches

    features = np.concatenate(
        all_features,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )


    print(
        f"{dataset_name} feature shape:",
        features.shape
    )

    print(
        f"{dataset_name} label shape:",
        labels.shape
    )

    return features, labels


# ============================================================
# 8. EXTRACT TRAINING FEATURES
# ============================================================

X_train, y_train = extract_dataset_features(
    train_loader,
    "Training"
)


# ============================================================
# 9. EXTRACT VALIDATION FEATURES
# ============================================================

X_val, y_val = extract_dataset_features(
    val_loader,
    "Validation"
)


# ============================================================
# 10. EXTRACT TEST FEATURES
# ============================================================

X_test, y_test = extract_dataset_features(
    test_loader,
    "Testing"
)


# ============================================================
# 11. SAVE FEATURES
# ============================================================

np.save(
    "X_train_features.npy",
    X_train
)

np.save(
    "y_train_features.npy",
    y_train
)

np.save(
    "X_val_features.npy",
    X_val
)

np.save(
    "y_val_features.npy",
    y_val
)

np.save(
    "X_test_features.npy",
    X_test
)

np.save(
    "y_test_features.npy",
    y_test
)


# ============================================================
# 12. VERIFY SAVED FEATURES
# ============================================================

print("\n" + "=" * 60)

print("FEATURE EXTRACTION COMPLETE")

print("=" * 60)

print(
    "Training features:",
    X_train.shape
)

print(
    "Validation features:",
    X_val.shape
)

print(
    "Testing features:",
    X_test.shape
)

print("\nSaved files:")

print("X_train_features.npy")
print("y_train_features.npy")

print("X_val_features.npy")
print("y_val_features.npy")

print("X_test_features.npy")
print("y_test_features.npy")

print("=" * 60)