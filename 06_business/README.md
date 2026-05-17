# 06 — Customer LTV & RFM Segmentation

**Business Question:** Who are our most valuable customers, how much are they worth, and what happens if we lose them?

---

## Overview

This project analyzes 12 months of real e-commerce transactions to segment customers by behavior, calculate their Lifetime Value, and quantify the financial impact of retention vs churn — translating data into actionable business decisions with £ figures attached.

**Dataset:** [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — 541,909 transactions from a UK-based gift retailer (Dec 2010 – Dec 2011)

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Load, clean, and explore the dataset. Revenue by country, time trends, Pareto analysis |
| `02_rfm_segmentation.ipynb` | RFM metric calculation, K-Means clustering, segment naming and profiling |
| `03_ltv_business_impact.ipynb` | LTV per segment, revenue at risk, retention ROI simulation, executive recommendations |

---

## Key Results

- **Pareto effect confirmed:** top 20% of customers drive ~80% of revenue
- **4 segments identified:** Champions, Loyal Customers, At-Risk, Lost
- **Retention ROI:** win-back campaign for At-Risk customers breaks even at ~5–10% conversion rate
- **Retention vs Acquisition:** retaining an existing customer costs 5x less than acquiring a new one

---

## Setup

```bash
pip install -r requirements.txt
```

**Download the dataset:**
1. Go to https://archive.ics.uci.edu/dataset/352/online+retail
2. Download `Online Retail.xlsx`
3. Place it in `data/raw/Online Retail.xlsx`

Then run the notebooks in order: `01` → `02` → `03`

---

## Tech Stack

- **Python** · pandas · numpy · scikit-learn · matplotlib · seaborn
- **ML:** K-Means Clustering, StandardScaler
- **Analysis:** RFM Framework, LTV Modeling, ROI Simulation

---

## Business Skills Demonstrated

- Translating raw transactions into strategic customer segments
- Calculating Customer Lifetime Value from actual behavior data
- Quantifying revenue at risk in monetary terms
- Building ROI models to support budget decisions
- Communicating data findings at executive level
