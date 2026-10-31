import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BrainTumorDataset


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("DEVICE:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 60)


# ============================================================
# 2. CONFIGURATION
# ============================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
NUM_CLASSES = 4

TRAIN_CSV = "train.csv"
VAL_CSV = "validation.csv"

BEST_MODEL_PATH = "best_cnn_model.pth"


# ============================================================
# 3. DATASETS
# ============================================================

train_dataset = BrainTumorDataset(TRAIN_CSV)
val_dataset = BrainTumorDataset(VAL_CSV)

print("\nDataset sizes:")
print("Training:", len(train_dataset))
print("Validation:", len(val_dataset))


# ============================================================
# 4. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 5. CUSTOM CNN
# ============================================================

class BrainTumorCNN(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        # ----------------------------------------------------
        # Convolutional Block 1
        # Input: 1 x 224 x 224
        # Output: 64 channels
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

            nn.MaxPool2d(kernel_size=2)
        )

        # ----------------------------------------------------
        # Convolutional Block 2
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

            nn.AvgPool2d(kernel_size=2)
        )

        # ----------------------------------------------------
        # Convolutional Block 3
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

            nn.MaxPool2d(kernel_size=2)
        )

        # ----------------------------------------------------
        # Convolutional Block 4
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

            nn.AvgPool2d(kernel_size=2)
        )

        # ----------------------------------------------------
        # Feature representation
        #
        # After four 2x2 pooling operations:
        #
        # 224 -> 112 -> 56 -> 28 -> 14
        #
        # Feature map:
        # 1024 x 14 x 14
        # ----------------------------------------------------

        self.flatten = nn.Flatten()

        self.feature_layer = nn.Sequential(
            nn.Linear(1024 * 14 * 14, 1024),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(p=0.50)

        # ----------------------------------------------------
        # Final classification layer
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            1024,
            num_classes
        )


    def forward(self, x):

        # CNN feature extraction

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Flatten feature map

        x = self.flatten(x)

        # Dense feature representation

        x = self.feature_layer(x)

        # Dropout

        x = self.dropout(x)

        # Four-class prediction

        x = self.classifier(x)

        return x


# ============================================================
# 6. CREATE MODEL
# ============================================================

model = BrainTumorCNN(
    num_classes=NUM_CLASSES
).to(device)

print("\nModel created successfully.")


# ============================================================
# 7. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 8. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 9. BEST MODEL TRACKING
# ============================================================

best_val_accuracy = 0.0


# ============================================================
# 10. TRAINING LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients

        optimizer.zero_grad()

        # Forward pass

        outputs = model(images)

        # Calculate loss

        loss = criterion(outputs, labels)

        # Backpropagation

        loss.backward()

        # Update weights

        optimizer.step()

        # Track loss

        running_loss += loss.item() * images.size(0)

        # Predictions

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


    train_loss = running_loss / len(train_dataset)
    train_accuracy = 100 * correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass

            outputs = model(images)

            # Validation loss

            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            # Predictions

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = val_running_loss / len(val_dataset)

    val_accuracy = 100 * val_correct / val_total


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy
            },
            BEST_MODEL_PATH
        )

        print(
            f"*** BEST MODEL SAVED "
            f"({val_accuracy:.2f}%) ***"
        )


# ============================================================
# 11. TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)

print("TRAINING COMPLETE")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best model saved as: "
    f"{BEST_MODEL_PATH}"
)

print("=" * 60)