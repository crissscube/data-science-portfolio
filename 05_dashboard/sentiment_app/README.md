# Sentiment Analyzer — Interactive Streamlit App

## What it does
Interactive web app that predicts whether a movie review is positive or negative in real time, showing confidence score and the words that most influenced the prediction.

## Live Demo
Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model
- **Algorithm:** Logistic Regression + TF-IDF (10,000 features, unigrams + bigrams)
- **Training data:** NLTK movie_reviews corpus (2,000 reviews)
- **Accuracy:** 83.5% | **AUC-ROC:** 0.919
- Model trains on startup (~5 seconds), no pickle files needed

## Features
- Real-time sentiment prediction with confidence score
- Highlights words that pushed the prediction positive or negative
- Example reviews to test instantly

## Tools
Python · Streamlit · scikit-learn · NLTK
