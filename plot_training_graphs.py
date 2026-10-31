
import matplotlib.pyplot as plt
import os

# Output directory for graphs
os.makedirs("training_graphs", exist_ok=True)

# Epochs
epochs = list(range(1, 11))

# Recorded CNN training history
train_accuracy = [
    70.04, 82.63, 86.60, 89.52, 91.22,
    93.05, 93.28, 94.66, 95.34, 95.69
]

val_accuracy = [
    85.00, 86.90, 88.69, 88.21, 91.79,
    91.79, 94.40, 94.29, 90.00, 95.83
]

train_loss = [
    1.4728, 0.4549, 0.3574, 0.2781, 0.2451,
    0.1933, 0.1830, 0.1489, 0.1216, 0.1206
]

val_loss = [
    0.4654, 0.3681, 0.3166, 0.3123, 0.2328,
    0.2156, 0.1740, 0.1795, 0.2825, 0.1585
]

# -----------------------------------------
# GRAPH 1: Training vs Validation Accuracy
# -----------------------------------------
plt.figure(figsize=(9, 6))

plt.plot(
    epochs, train_accuracy,
    marker="o", linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    epochs, val_accuracy,
    marker="s", linewidth=2,
    label="Validation Accuracy"
)

plt.title("CNN Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(epochs)
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(
    "training_graphs/figure_6_2_accuracy.png",
    dpi=300, bbox_inches="tight"
)
plt.show()

# -----------------------------------------
# GRAPH 2: Training vs Validation Loss
# -----------------------------------------
plt.figure(figsize=(9, 6))

plt.plot(
    epochs, train_loss,
    marker="o", linewidth=2,
    label="Training Loss"
)

plt.plot(
    epochs, val_loss,
    marker="s", linewidth=2,
    label="Validation Loss"
)

plt.title("CNN Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(
    "training_graphs/figure_6_3_loss.png",
    dpi=300, bbox_inches="tight"
)
plt.show()

print("Both graphs generated successfully!")
print("Saved inside the training_graphs folder.")
