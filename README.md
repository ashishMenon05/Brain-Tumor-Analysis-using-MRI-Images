# Brain Tumor Classification and Analysis Using MRI Images

**Team Tech Techerz** — Final Year Project  
**Sri Ramakrishna Institute of Technology**  
Department of Computer Science and Engineering

| Team Member | Role |
|---|---|
| Ashish M Menon | CNN Training, Feature Extraction |
| Chandhini SJ | Dataset, Preprocessing, CNN Architecture |
| Janani E | SVM Classification, Results & Evaluation |
| Hussain Mustafa Ali | Comparison, Limitations, Conclusion |

---

## 📌 Project Overview

This project develops and evaluates a hybrid deep learning + machine learning pipeline for classifying brain MRI images into four categories:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

Two approaches are implemented and compared:
1. **Custom CNN** — end-to-end image classification using a 4-block Convolutional Neural Network built in PyTorch.
2. **Hybrid CNN + SVM** — the trained CNN is used as a feature extractor, and an SVM classifier is applied to the 1,024-dimensional feature vectors.

---

## 📂 Project Structure

```
FINAL YEAR PROJECT/
│
├── Training/                  # Training images (5,600 images, 4 classes)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── Testing/                   # Test images (1,600 images, 4 classes)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── Papers/                    # Reference research papers (PDFs)
│
├── train.csv                  # Training set file paths and labels (4,760 images)
├── validation.csv             # Validation set file paths and labels (840 images)
├── test.csv                   # Test set file paths and labels (1,600 images)
│
├── dataset.py                 # PyTorch Dataset class
├── model.py                   # CNN model definition (standalone)
├── preprocess.py              # Image preprocessing utilities
│
├── build_dataset.py           # Builds train/val/test CSV splits
├── dataset_inspection.py      # Inspect dataset class distribution
├── view_images.py             # Visualize sample MRI images
│
├── train.py                   # CNN training script → saves best_cnn_model.pth
├── extract_features.py        # CNN feature extraction → saves .npy feature arrays
├── svm_classifier.py          # SVM training using CNN features
├── svm_tuning.py              # SVM hyperparameter tuning (GridSearchCV)
│
├── evaluate.py                # CNN model evaluation (test set)
├── final_svm_evaluation.py    # CNN-SVM evaluation (test set)
│
├── plot_training_graphs.py    # Plot training accuracy and loss curves
├── plot_model_comparison.py   # Plot CNN vs CNN-SVM comparison bar chart
│
├── preprocessing_demo.py      # Demonstrate preprocessing pipeline visually
├── test_dataloader.py         # Test the DataLoader works correctly
├── test_gpu.py                # Check CUDA GPU availability
├── test_model.py              # Quick model forward-pass test
├── test_preprocess.py         # Test preprocessing on a single image
│
└── training_graphs/           # Generated training plots (after running code)
```

---

## 🧠 Dataset

**Brain Tumor MRI Dataset** by Masoud Nickparvar — available on [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Split | Images |
|---|---|
| Training | 4,760 |
| Validation | 840 |
| Testing | 1,600 |
| **Total** | **7,200** |

---

## ⚙️ Preprocessing Pipeline

1. Convert to **grayscale** (single channel)
2. Apply **Gaussian filter** (σ = 1.5) to reduce noise
3. **Resize** to 224 × 224 pixels
4. **Normalize** pixel values to [0, 1] (divide by 255)

---

## 🏗️ CNN Architecture

A custom 4-block CNN built with **PyTorch**:

| Block | Layer | Output Channels | Pooling |
|---|---|---|---|
| Block 1 | Conv2d + BN + ReLU | 64 | MaxPool 2×2 |
| Block 2 | Conv2d + BN + ReLU | 128 | AvgPool 2×2 |
| Block 3 | Conv2d + BN + ReLU | 256 | MaxPool 2×2 |
| Block 4 | Conv2d + BN + ReLU | 1,024 | AvgPool 2×2 |
| Feature Layer | Linear(200,704 → 1,024) + ReLU | — | — |
| Dropout | p = 0.5 | — | — |
| Classifier | Linear(1,024 → 4) | 4 classes | — |

**Training settings:**
- Optimizer: Adam (lr = 1e-4)
- Loss: CrossEntropyLoss
- Batch Size: 32
- Epochs: 10

---

## 📊 Results

| Model | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Custom CNN | 91.25% | 91.71% | 91.25% | 91.04% |
| **CNN + SVM (Hybrid)** | **91.50%** | **91.89%** | **91.50%** | **91.35%** |

- **Best CNN Validation Accuracy:** 95.83% (Epoch 10)
- **Best SVM Configuration:** RBF kernel, C = 1.0, gamma = scale
- **Best SVM Validation Accuracy:** 96.19%

---

## 🚀 How to Run

### Prerequisites

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn pandas pillow scipy
```

### Step-by-Step Pipeline

```bash
# Prerequisites installation
pip install -r requirements.txt

# Option A: Run the Interactive Web Dashboard (Recommended)
python app.py
# Open http://localhost:5000 in your browser to train, monitor, and classify MRI scans interactively.

# Option B: Run via CLI step-by-step
# Step 1: Verify GPU availability
python test_gpu.py

# Step 2: Build dataset CSV splits (if not already present)
python build_dataset.py

# Step 3: Train the CNN model (saves best_cnn_model.pth)
python train.py

# Step 4: Extract CNN features (saves .npy arrays)
python extract_features.py

# Step 5: Train SVM classifier on CNN features
python svm_classifier.py

# Step 6: (Optional) Tune SVM hyperparameters
python svm_tuning.py

# Step 7: Evaluate CNN model on test set
python evaluate.py

# Step 8: Evaluate CNN-SVM hybrid on test set
python final_svm_evaluation.py

# Step 9: Generate training graphs
python plot_training_graphs.py

# Step 10: Generate model comparison plot
python plot_model_comparison.py
```

---

## 📦 Generated Files (after running pipeline)

| File | Description |
|---|---|
| `best_cnn_model.pth` | Best CNN checkpoint (~2.5 GB) |
| `cnn_feature_svm.pkl` | Trained SVM model |
| `best_cnn_feature_svm.pkl` | Best tuned SVM model |
| `X_train_features.npy` | Training CNN features (4760 × 1024) |
| `X_val_features.npy` | Validation CNN features (840 × 1024) |
| `X_test_features.npy` | Test CNN features (1600 × 1024) |
| `y_train/val/test_features.npy` | Corresponding labels |
| `cnn_confusion_matrix.png` | CNN confusion matrix |
| `svm_confusion_matrix.png` | SVM confusion matrix |
| `final_svm_confusion_matrix.png` | Final SVM evaluation confusion matrix |
| `training_graphs/*.png` | Accuracy/loss curves and comparison plots |

---

## ⚠️ Disclaimer

This project is an **academic research prototype** developed as part of a final-year undergraduate project. It has **not undergone clinical validation** and must **not be used as a substitute for professional medical diagnosis**.

---

## 📄 License

This project is submitted for academic evaluation at Sri Ramakrishna Institute of Technology.  
Dataset credit: [Masoud Nickparvar on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
