# Skin Cancer Detection using TensorFlow & EfficientNet

A deep learning project that classifies skin lesion images as **malignant** or **benign** using transfer learning with EfficientNetB7 and TensorFlow/Keras.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)

---

## Overview

Skin cancer is one of the most common and dangerous forms of cancer. Early detection is critical for effective treatment. This project uses a **Convolutional Neural Network (CNN)** powered by **EfficientNetB7** (pre-trained on ImageNet) to distinguish between malignant and benign skin lesions from dermoscopic images.

The model leverages **transfer learning** — reusing the feature extraction power of a large pretrained network — to achieve high performance with limited training data and time.

---

## Dataset

- **Source:** [Kaggle - Skin Cancer ISIC Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)
- **Total Images:** ~2,637
- **Classes:**
  - `malignant` (~45.4%)
  - `benign` (~54.6%)
- **Format:** `.jpg` images organized in subdirectories by class

The dataset is nearly balanced, so class imbalance is not a concern.

>  The dataset is **not included** in this repository due to file size. Please download it from Kaggle and place it as described in [Project Structure](#project-structure).

---

## Project Structure

```
skin_cancer_detection/
│
├── train_cancer/
│   ├── malignant/          # Malignant skin lesion images
│   └── benign/             # Benign skin lesion images
│
├── skin_cancer.ipynb       # Main Jupyter Notebook with all code
└── README.md               # Project documentation
```

---


## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/skin-cancer-detection.git
cd skin-cancer-detection
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)
2. Download and extract the zip file
3. Place images into the following folders:
   - `train_cancer/malignant/` — malignant lesion images
   - `train_cancer/benign/` — benign lesion images

---

##  Usage

1. Open the project folder in **VS Code**
2. Open `skin_cancer.ipynb`
3. Select your virtual environment as the Jupyter kernel (top-right of the notebook)
4. Run all cells using **Run All** or **Shift + Enter** cell by cell

---

##  Model Architecture

The model is built using the **Keras Functional API** with the following components:

| Layer | Details |
|---|---|
| Input | (224, 224, 3) — resized RGB images |
| Base Model | EfficientNetB7 (pre-trained on ImageNet, frozen) |
| Flatten | Flattens base model output |
| Dense | 256 units, ReLU activation |
| BatchNormalization | Stabilizes training |
| Dense | 256 units, ReLU activation |
| Dropout | Rate = 0.3 (prevents overfitting) |
| BatchNormalization | Stabilizes training |
| Output | 1 unit, Sigmoid activation (binary classification) |

### Key Design Choices

- **Transfer Learning:** EfficientNetB7 weights are frozen — only the custom head is trained. This dramatically reduces training time while leveraging powerful pretrained features.
- **Dropout:** Added before the final layer to reduce overfitting.
- **BatchNormalization:** Ensures stable and fast convergence.
- **Binary Cross-Entropy Loss:** Appropriate for two-class classification.
- **Adam Optimizer:** Adaptive learning rate for efficient training.
- **AUC Metric:** Area Under the ROC Curve — a robust metric for imbalanced or medical datasets.

---

## 📊 Results

The model was trained for **5 epochs** on ~2,241 training images and validated on ~396 images.

| Epoch | Train Loss | Train AUC | Val Loss | Val AUC |
|---|---|---|---|---|
| 1 | 0.5478 | 0.8139 | 2.6825 | 0.6711 |
| 2 | 0.4547 | 0.8674 | 1.1363 | 0.8328 |
| 3 | 0.4288 | 0.8824 | 0.8702 | 0.8385 |
| 4 | 0.4044 | 0.8933 | 0.6367 | 0.8561 |
| 5 | 0.3891 | 0.9019 | 0.9296 | 0.8558 |

- Final **Train AUC: ~0.90**
- Final **Validation AUC: ~0.856**

> Training loss decreased steadily across epochs. Validation loss showed some fluctuation, suggesting potential for improvement with more epochs or data augmentation.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10 | Core programming language |
| TensorFlow / Keras | Deep learning framework |
| EfficientNetB7 | Pre-trained base model (transfer learning) |
| NumPy | Numerical operations |
| Pandas | Data manipulation and DataFrames |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Train-test split |
| Pillow (PIL) | Image loading and processing |
| Glob | File path traversal |

---

##  Author

**Your Name**
- GitHub: [@manisha3389293](https://github.com/manisha3389293)

---
