# Machine Learning — Employee Attrition Prediction

## Business Problem
Can we predict which employees are likely to leave before they do?
This project builds a classification model on the IBM HR Analytics dataset to identify high-risk employees, enabling HR to take proactive retention actions.

## Dataset
Same dataset as `01_eda/` — IBM HR Analytics (1,470 employees · 35 features)

## Approach
1. **Preprocessing** — encoding, scaling, handling class imbalance
2. **Modeling** — Logistic Regression (baseline) + Random Forest (main model)
3. **Evaluation** — ROC-AUC, precision, recall, confusion matrix
4. **Feature importance** — which variables drive the predictions

## Notebooks
| Notebook | Description |
|----------|-------------|
| `01_preprocessing.ipynb` | Feature engineering, encoding, train/test split |
| `02_modeling.ipynb` | Model training, evaluation, feature importance |

## Results
*(To be completed after running the notebooks)*

## Setup
```bash
pip install -r requirements.txt
```

## Tools
Python · pandas · scikit-learn · matplotlib · seaborn
