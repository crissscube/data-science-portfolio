# EDA — IBM HR Analytics: Employee Attrition

## Business Problem
Employee turnover is expensive. Replacing one employee can cost between 50% and 200% of their annual salary.
This project explores the IBM HR Analytics dataset to identify which factors are most associated with employees leaving the company.

## Dataset
- **Source:** [IBM HR Analytics on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Rows:** 1,470 employees
- **Columns:** 35 features (demographics, job info, satisfaction scores)
- **Target:** `Attrition` (Yes / No)

## Key Questions
1. What is the overall attrition rate?
2. Which departments and job roles have the highest turnover?
3. Does salary influence attrition?
4. Do overtime and work-life balance play a role?
5. What employee profile is most at risk of leaving?

## Main Findings

1. **Overall attrition rate is ~16%.** 237 of 1,470 employees left the company — a costly figure given replacement costs of 50–200% of annual salary.
2. **Overtime is the strongest driver.** Employees who work overtime leave at **30.5%** vs **10.4%** for those who don't — roughly **3x higher**. This is the single most actionable lever for HR.
3. **Lower income drives turnover.** Employees who left earned **~30% less** on average (≈$4,787 vs $6,833 monthly).
4. **Early-career employees are most at risk.** Those who left are younger (avg. 34 vs 38) and have fewer years at the company (avg. 5.1 vs 7.4).
5. **Some roles are critical.** Sales Representatives (**39.8%**) and Laboratory Technicians (**23.9%**) show the highest role-specific attrition.

**Recommended actions:** limit/compensate mandatory overtime, review salaries for the bottom earners, and add mentorship and clear career paths for early-career and high-turnover roles.

> Full analysis, visualizations, and a business-recommendations table are in the notebook.

## Setup
```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it at: `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`

## Tools
Python · pandas · matplotlib · seaborn
