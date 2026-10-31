import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD CNN FEATURES
# ============================================================

print("=" * 60)
print("LOADING CNN FEATURES")
print("=" * 60)

X_train = np.load("X_train_features.npy")
y_train = np.load("y_train_features.npy")

X_val = np.load("X_val_features.npy")
y_val = np.load("y_val_features.npy")


print("\nTraining features:", X_train.shape)
print("Training labels:", y_train.shape)

print("Validation features:", X_val.shape)
print("Validation labels:", y_val.shape)


# ============================================================
# 2. SVM CONFIGURATIONS
# ============================================================

configs = [

    # RBF kernel
    {
        "kernel": "rbf",
        "C": 0.1,
        "gamma": "scale"
    },

    {
        "kernel": "rbf",
        "C": 1,
        "gamma": "scale"
    },

    {
        "kernel": "rbf",
        "C": 10,
        "gamma": "scale"
    },

    {
        "kernel": "rbf",
        "C": 100,
        "gamma": "scale"
    },

    # Linear kernel
    {
        "kernel": "linear",
        "C": 0.1
    },

    {
        "kernel": "linear",
        "C": 1
    },

    {
        "kernel": "linear",
        "C": 10
    }
]


# ============================================================
# 3. RESULTS STORAGE
# ============================================================

results = []

best_model = None
best_config = None
best_f1 = -1


# ============================================================
# 4. TRAIN AND VALIDATE
# ============================================================

print("\n" + "=" * 60)
print("SVM VALIDATION EXPERIMENTS")
print("=" * 60)


for index, config in enumerate(configs):

    print(
        f"\nExperiment {index + 1}/{len(configs)}"
    )

    print(
        "Kernel:",
        config["kernel"]
    )

    print(
        "C:",
        config["C"]
    )

    if "gamma" in config:

        print(
            "Gamma:",
            config["gamma"]
        )


    # --------------------------------------------------------
    # Create SVM
    # --------------------------------------------------------

    svm = SVC(
        kernel=config["kernel"],
        C=config["C"],
        gamma=config.get("gamma", "scale"),
        probability=False,
        random_state=42
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    svm.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    val_predictions = svm.predict(
        X_val
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    precision = precision_score(
        y_val,
        val_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        average="macro",
        zero_division=0
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Validation Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Validation Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Validation Recall   : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Validation F1       : "
        f"{f1 * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({
        "kernel": config["kernel"],
        "C": config["C"],
        "gamma": config.get("gamma", "scale"),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


    # --------------------------------------------------------
    # Track best model
    #
    # We use validation F1 as the selection criterion.
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1

        best_model = svm

        best_config = config.copy()


# ============================================================
# 5. PRINT COMPARISON
# ============================================================

print("\n" + "=" * 80)

print("SVM VALIDATION COMPARISON")

print("=" * 80)

print(
    f"{'Kernel':<10}"
    f"{'C':<10}"
    f"{'Gamma':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 80)


for result in results:

    print(
        f"{result['kernel']:<10}"
        f"{str(result['C']):<10}"
        f"{str(result['gamma']):<12}"
        f"{result['accuracy'] * 100:<12.2f}"
        f"{result['precision'] * 100:<12.2f}"
        f"{result['recall'] * 100:<12.2f}"
        f"{result['f1'] * 100:<12.2f}"
    )


# ============================================================
# 6. BEST CONFIGURATION
# ============================================================

print("\n" + "=" * 80)

print("BEST SVM CONFIGURATION")

print("=" * 80)

print(
    "Kernel:",
    best_config["kernel"]
)

print(
    "C:",
    best_config["C"]
)

print(
    "Gamma:",
    best_config.get("gamma", "scale")
)

print(
    f"Best Validation F1: "
    f"{best_f1 * 100:.2f}%"
)

print("=" * 80)


# ============================================================
# 7. SAVE BEST SVM
# ============================================================

import joblib

joblib.dump(
    best_model,
    "best_cnn_feature_svm.pkl"
)

print(
    "\nBest SVM saved as:"
)

print(
    "best_cnn_feature_svm.pkl"
)

print(
    "\nSVM tuning completed."
)