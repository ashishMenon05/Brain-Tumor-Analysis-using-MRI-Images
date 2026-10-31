import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BrainTumorDataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import numpy as np
import matplotlib.pyplot as plt


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

TEST_CSV = "test.csv"

MODEL_PATH = "best_cnn_model.pth"

BATCH_SIZE = 32

NUM_CLASSES = 4

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ============================================================
# 3. TEST DATASET
# ============================================================

test_dataset = BrainTumorDataset(
    TEST_CSV
)

print("\nTest dataset size:", len(test_dataset))


# ============================================================
# 4. TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 5. CNN MODEL
# ============================================================

class BrainTumorCNN(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        # ----------------------------------------------------
        # Block 1
        # ----------------------------------------------------

        self.block1 = nn.Sequential(
            nn.Conv2d(
                1,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        # ----------------------------------------------------
        # Block 2
        # ----------------------------------------------------

        self.block2 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2)
        )

        # ----------------------------------------------------
        # Block 3
        # ----------------------------------------------------

        self.block3 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        # ----------------------------------------------------
        # Block 4
        # ----------------------------------------------------

        self.block4 = nn.Sequential(
            nn.Conv2d(
                256,
                1024,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2)
        )

        # ----------------------------------------------------
        # Dense feature representation
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
        # Classification layer
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            1024,
            num_classes
        )


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


# ============================================================
# 6. CREATE MODEL
# ============================================================

model = BrainTumorCNN(
    num_classes=NUM_CLASSES
).to(device)


# ============================================================
# 7. LOAD BEST CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("\nBest checkpoint loaded.")

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    f"Checkpoint validation accuracy: "
    f"{checkpoint['val_accuracy']:.2f}%"
)


# ============================================================
# 8. EVALUATION
# ============================================================

model.eval()

all_labels = []
all_predictions = []

print("\nEvaluating untouched test set...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# Convert to NumPy arrays

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)


# ============================================================
# 9. METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)

print("CNN TEST RESULTS")

print("=" * 60)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-score  : {f1 * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nCONFUSION MATRIX")

print(cm)


# ============================================================
# 13. PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "CNN Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

# Add values inside cells

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "cnn_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. COMPLETE
# ============================================================

print("\nConfusion matrix saved as:")

print(
    "cnn_confusion_matrix.png"
)

print("\nCNN evaluation completed.")