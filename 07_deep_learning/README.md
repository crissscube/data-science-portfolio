# 07 — Digit Recognition with CNN (PyTorch)

**Business Problem:** Automate handwritten digit recognition in physical documents — bank checks, utility meters, paper forms — replacing manual data entry with a model that achieves 99%+ accuracy.

---

## Overview

A Convolutional Neural Network trained end-to-end with PyTorch on the MNIST dataset (70,000 images). Includes full training pipeline, error analysis, confidence calibration, and filter visualization.

**Dataset:** MNIST — downloaded automatically via `torchvision`

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_cnn_training.ipynb` | Data loading, augmentation, CNN architecture, training loop, learning curves, test evaluation |
| `02_error_analysis.ipynb` | Per-class accuracy, misclassified examples, confidence analysis, filter visualization, business conclusions |

---

## Architecture

```
Input (1×28×28)
  ├── Conv Block 1: Conv2d(1→32) + BatchNorm + ReLU + MaxPool → 32×14×14
  ├── Conv Block 2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool → 64×7×7
  ├── Conv Block 3: Conv2d(64→128) + BatchNorm + ReLU + MaxPool → 128×3×3
  └── Classifier: Linear(1152→256) → ReLU → Dropout(0.5) → Linear(256→10)
```

~310,000 trainable parameters · Adam optimizer · ReduceLROnPlateau scheduler

---

## Results

- **Test Accuracy:** ~99%+
- **Training time:** ~5 min on CPU
- **Calibration:** High-confidence predictions are correct ~99% of the time

---

## Setup

```bash
pip install -r requirements.txt
```

Run notebooks in order: `01` → `02`  
The dataset downloads automatically on first run (~11 MB).

---

## Tech Stack

- **PyTorch** · torchvision · numpy · matplotlib · seaborn · scikit-learn
- **CNN** with BatchNorm, Dropout, MaxPool
- **Data augmentation:** RandomRotation + RandomAffine
- **LR scheduling:** ReduceLROnPlateau
