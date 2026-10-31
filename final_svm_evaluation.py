import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "best_cnn_feature_svm.pkl"

TEST_FEATURES = "X_test_features.npy"
TEST_LABELS = "y_test_features.npy"

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ============================================================
# 2. LOAD TEST FEATURES
# ============================================================

print("=" * 60)
print("FINAL CNN FEATURE + SVM EVALUATION")
print("=" * 60)

X_test = np.load(
    TEST_FEATURES
)

y_test = np.load(
    TEST_LABELS
)

print(
    "\nTest features:",
    X_test.shape
)

print(
    "Test labels:",
    y_test.shape
)


# ============================================================
# 3. LOAD BEST SVM
# ============================================================

svm = joblib.load(
    MODEL_PATH
)

print(
    "\nBest SVM loaded:"
)

print(
    "Kernel: RBF"
)

print(
    "C: 1"
)

print(
    "Gamma: scale"
)


# ============================================================
# 4. FINAL TEST PREDICTION
# ============================================================

print(
    "\nEvaluating on untouched test set..."
)

test_predictions = svm.predict(
    X_test
)


# ============================================================
# 5. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# 6. RESULTS
# ============================================================

print("\n" + "=" * 60)

print("FINAL TEST RESULTS")

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
# 7. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\nCONFUSION MATRIX")

print(cm)


# ============================================================
# 9. SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Final CNN Feature + SVM Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

for i in range(len(CLASS_NAMES)):

    for j in range(len(CLASS_NAMES)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "final_svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("FINAL SVM EVALUATION COMPLETE")

print("=" * 60)

print(
    f"Final Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Final Test F1-score: "
    f"{f1 * 100:.2f}%"
)

print(
    "\nConfusion matrix saved as:"
)

print(
    "final_svm_confusion_matrix.png"
)

print("=" * 60)