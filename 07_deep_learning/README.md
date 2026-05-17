# 07 — Digit Recognition with CNN (PyTorch)

**Business Problem:** Automate handwritten digit recognition in physical documents — bank checks, utility meters, paper forms — replacing manual data entry with a model that achieves 99.5%+ accuracy served via a production-ready REST API.

---

## Overview

A Convolutional Neural Network trained end-to-end with PyTorch on the MNIST dataset (70,000 images), deployed as a REST API inside a Docker container. Includes full training pipeline, error analysis, confidence calibration, filter visualization, and production deployment.

**Dataset:** MNIST — downloaded automatically via `torchvision`

---

## Structure

```
07_deep_learning/
├── 01_cnn_training.ipynb    # Architecture, training loop, curves, evaluation
├── 02_error_analysis.ipynb  # Error analysis, confidence, filter visualization
└── api/
    ├── main.py              # FastAPI server
    ├── model.py             # Model definition + inference logic
    ├── Dockerfile           # Docker container definition
    ├── requirements.txt     # API dependencies
    └── client_test.py       # Test client against MNIST test set
```

---

## Model Architecture

```
Input (1×28×28)
  ├── Conv Block 1: Conv2d(1→32) + BatchNorm + ReLU + MaxPool → 32×14×14
  ├── Conv Block 2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool → 64×7×7
  ├── Conv Block 3: Conv2d(64→128) + BatchNorm + ReLU + MaxPool → 128×3×3
  └── Classifier: Linear(1152→256) → ReLU → Dropout(0.5) → Linear(256→10)
```

~310,000 trainable parameters · Adam optimizer · ReduceLROnPlateau scheduler · Data augmentation

---

## Results

- **Test Accuracy:** 99.5%+
- **Training time:** ~15 min on CPU
- **API response:** < 100ms per prediction

---

## Run the API with Docker

```bash
# From the 07_deep_learning/ directory
docker build -f api/Dockerfile -t digit-api .
docker run -p 8000:8000 digit-api
```

API available at `http://localhost:8000`

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Model info |
| GET | `/health` | Health check |
| POST | `/predict` | Predict digit from image |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@digit.png"
```

**Example response:**
```json
{
  "filename": "digit.png",
  "predicted_digit": 7,
  "confidence": "99.87%",
  "all_probabilities": {"0": 0.0, "1": 0.0, "7": 0.9987, ...}
}
```

**Test against MNIST:**
```bash
python api/client_test.py
```

---

## Setup (training only)

```bash
pip install -r requirements.txt
```

Run notebooks in order: `01` → `02`. Dataset downloads automatically (~11 MB).

---

## Tech Stack

**Training:** PyTorch · torchvision · numpy · matplotlib · seaborn · scikit-learn

**Deployment:** FastAPI · Docker · uvicorn · Pillow
