
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output folder
os.makedirs("training_graphs", exist_ok=True)

# Evaluation metrics (%)
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

cnn_scores = [91.25, 91.71, 91.25, 91.04]
cnn_svm_scores = [91.50, 91.89, 91.50, 91.35]

# Bar chart positions
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

# Plot bars
bars1 = plt.bar(
    x - width / 2,
    cnn_scores,
    width,
    label="Custom CNN"
)

bars2 = plt.bar(
    x + width / 2,
    cnn_svm_scores,
    width,
    label="CNN + SVM"
)

# Add values above bars
plt.bar_label(bars1, fmt="%.2f%%", padding=3)
plt.bar_label(bars2, fmt="%.2f%%", padding=3)

# Chart formatting
plt.title("CNN vs CNN-SVM Performance Comparison")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score (%)")
plt.xticks(x, metrics)
plt.ylim(85, 100)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

# Save high-resolution image
output_path = "training_graphs/cnn_vs_cnn_svm_comparison.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Chart saved to: {output_path}")
