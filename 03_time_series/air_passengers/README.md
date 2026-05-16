# Time Series — Air Passengers Forecasting

## Business Problem
Airlines need to forecast passenger demand to plan capacity, staffing, and fuel. This project analyzes monthly passenger data and builds a SARIMA model to predict future demand.

## Dataset
- **Source:** Built-in dataset from `statsmodels` library
- **Period:** January 1949 – December 1960 (144 months)
- **Variable:** Monthly total international airline passengers (thousands)

## Key Questions
1. What are the trend and seasonality patterns in air travel?
2. Is the series stationary? What transformations are needed?
3. What SARIMA parameters best fit the data?
4. How accurate is the 12-month forecast?

## Notebooks
| Notebook | Description |
|----------|-------------|
| `01_eda_decomposition.ipynb` | Visualization, stationarity tests, decomposition |
| `02_arima_sarima.ipynb` | ARIMA/SARIMA modeling, forecasting, evaluation |

## Results
*(To be completed after running the notebooks)*

## Setup
```bash
pip install -r requirements.txt
```
No data download needed — dataset is included in `statsmodels`.

## Tools
Python · pandas · statsmodels · matplotlib · seaborn
