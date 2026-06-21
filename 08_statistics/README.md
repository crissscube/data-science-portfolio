# Statistics for Data Science — A Practical Guide

**Goal:** Demonstrate the core statistical toolkit every data scientist relies on,
applied end-to-end on a real dataset. Each technique is explained conceptually,
implemented in code, and interpreted from a **business perspective** — the
recurring theme being that translating results into clear, decision-ready
statements is what turns analysis into impact.

---

## Overview

A single, self-contained notebook that walks through descriptive and inferential
statistics using the Titanic dataset as a sandbox. The focus is on **reasoning**:
when to use each method, how to read its output, and how to explain it to a
non-technical audience.

**Dataset:** Titanic — 891 passengers (loaded directly from a public URL, no
manual download required).

---

## Contents

| # | Topic | What it answers |
|---|-------|-----------------|
| 1 | Setup & Data Loading | Cleaning and preparing the data |
| 2 | Distributions & the Normal curve | Is this value typical or extreme? |
| 3 | Z-scores & Outlier Detection | How far is a value from the mean? |
| 4 | Correlation Analysis | Which variables move together? |
| 5 | The Binomial Distribution | How likely are *k* successes in *n* trials? |
| 6 | Hypothesis Testing (t-test) | Is the difference between two groups real? |
| 7 | Confidence Intervals & A/B Testing | How precise is my estimate? |
| 8 | ANOVA + Tukey HSD | Which of several groups differ? |
| 9 | Linear Regression | How much does each driver affect a number? |
| 10 | Logistic Regression & Odds Ratios | Which factors drive a yes/no outcome, and by how much? |
| 11 | Bootstrap | How uncertain is *any* statistic, assumption-free? |
| 12 | Bayesian Thinking | How should evidence update my beliefs? |
| 13 | Conclusions | Summary of the toolkit |

---

## Selected Insights

- Women had roughly **13x higher odds of survival** than men (logistic regression,
  controlling for class and age).
- Mean fare (~$32) is **more than double** the median (~$14) — a textbook case of
  right-skewed data where the median is the honest summary.
- First class is significantly more expensive than second and third (ANOVA +
  Tukey), but second and third class are statistically indistinguishable in price.

---

## Setup

```bash
pip install -r requirements.txt
```

The dataset loads automatically from a public URL — no download needed.

## Tools

Python · pandas · NumPy · SciPy · statsmodels · matplotlib · seaborn
