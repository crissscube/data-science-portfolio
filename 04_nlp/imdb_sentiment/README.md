# NLP — IMDB Sentiment Analysis

## Business Problem
Automatically classify movie reviews as positive or negative using Natural Language Processing. This type of model is used in production systems to analyze customer feedback, social media, and product reviews at scale.

## Dataset
- **Source:** NLTK `movie_reviews` corpus (built-in, no download required)
- **Size:** 2,000 reviews — 1,000 positive, 1,000 negative
- **Balance:** Perfectly balanced (50/50)

## Approach
1. Text cleaning — remove HTML, punctuation, stopwords, lemmatization
2. TF-IDF vectorization — 10,000 features, unigrams + bigrams
3. Three classifiers compared: Logistic Regression, Naive Bayes, Random Forest

## Results

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| **Logistic Regression** | **83.50%** | **0.919** |
| Random Forest | 83.00% | 0.917 |
| Naive Bayes | 81.00% | 0.897 |

## Setup
```bash
pip install -r requirements.txt
```

## Tools
Python · pandas · NLTK · scikit-learn · WordCloud · matplotlib · seaborn
