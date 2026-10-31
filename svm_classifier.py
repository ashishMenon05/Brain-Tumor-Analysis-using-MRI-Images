import numpy as np

from sklearn.svm import SVC

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

TRAIN_FEATURES = "X_train_features.npy"
TRAIN_LABELS = "y_train_features.npy"

VAL_FEATURES = "X_val_features.npy"
VAL_LABELS = "y_val_features.npy"

TEST_FEATURES = "X_test_features.npy"
TEST_LABELS = "y_test_features.npy"


CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ============================================================
# 2. LOAD FEATURES
# ============================================================

print("=" * 60)
print("LOADING CNN FEATURES")
print("=" * 60)


X_train = np.load(
    TRAIN_FEATURES
)

y_train = np.load(
    TRAIN_LABELS
)

X_val = np.load(
    VAL_FEATURES
)

y_val = np.load(
    VAL_LABELS
)

X_test = np.load(
    TEST_FEATURES
)

y_test = np.load(
    TEST_LABELS
)


print("\nFeature shapes:")

print(
    "Training:",
    X_train.shape,
    y_train.shape
)

print(
    "Validation:",
    X_val.shape,
    y_val.shape
)

print(
    "Testing:",
    X_test.shape,
    y_test.shape
)


# ============================================================
# 3. CREATE SVM
# ============================================================

print("\n" + "=" * 60)
print("CREATING SVM")
print("=" * 60)

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

print("Kernel: RBF")
print("C: 1.0")
print("Gamma: scale")


# ============================================================
# 4. TRAIN SVM
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SVM")
print("=" * 60)

svm.fit(
    X_train,
    y_train
)

print("SVM training completed.")


# ============================================================
# 5. VALIDATION PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION EVALUATION")
print("=" * 60)

val_predictions = svm.predict(
    X_val
)


val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_precision = precision_score(
    y_val,
    val_predictions,
    average="macro",
    zero_division=0
)

val_recall = recall_score(
    y_val,
    val_predictions,
    average="macro",
    zero_division=0
)

val_f1 = f1_score(
    y_val,
    val_predictions,
    average="macro",
    zero_division=0
)


print(
    f"Validation Accuracy  : "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Validation Precision : "
    f"{val_precision * 100:.2f}%"
)

print(
    f"Validation Recall    : "
    f"{val_recall * 100:.2f}%"
)

print(
    f"Validation F1-score  : "
    f"{val_f1 * 100:.2f}%"
)


# ============================================================
# 6. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_predictions = svm.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)


print(
    f"Test Accuracy  : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Precision : "
    f"{test_precision * 100:.2f}%"
)

print(
    f"Test Recall    : "
    f"{test_recall * 100:.2f}%"
)

print(
    f"Test F1-score  : "
    f"{test_f1 * 100:.2f}%"
)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("SVM CLASSIFICATION REPORT")
print("=" * 60)

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


print("\n" + "=" * 60)
print("SVM CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 9. PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "CNN Feature + SVM Confusion Matrix"
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


# Add numbers to cells

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
    "svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. SAVE SVM MODEL
# ============================================================

import joblib

joblib.dump(
    svm,
    "cnn_feature_svm.pkl"
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("SVM PIPELINE COMPLETE")

print("=" * 60)

print(
    f"CNN + SVM Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"CNN + SVM Test F1-score: "
    f"{test_f1 * 100:.2f}%"
)

print(
    "\nSVM model saved as:"
)

print(
    "cnn_feature_svm.pkl"
)

print(
    "\nConfusion matrix saved as:"
)

print(
    "svm_confusion_matrix.png"
)

print("=" * 60)