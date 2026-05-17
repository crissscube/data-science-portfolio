import streamlit as st
import re
import nltk
import numpy as np

from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ── Model training (cached — runs only once) ──────────────────────────────────
@st.cache_resource
def load_model():
    nltk.download('movie_reviews', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean(text):
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        tokens = text.split()
        return ' '.join(
            lemmatizer.lemmatize(w)
            for w in tokens if w not in stop_words and len(w) > 2
        )

    reviews, labels = [], []
    for fileid in movie_reviews.fileids():
        reviews.append(clean(movie_reviews.raw(fileid)))
        labels.append(1 if fileid.startswith('pos') else 0)

    tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=5)
    X = tfidf.fit_transform(reviews)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X, labels)

    return model, tfidf, clean

model, tfidf, clean = load_model()

# ── Helper: top influential words ─────────────────────────────────────────────
def get_top_words(text_clean, n=5):
    vec = tfidf.transform([text_clean]).toarray()[0]
    coef = model.coef_[0]
    scores = vec * coef
    top_pos_idx = scores.argsort()[-n:][::-1]
    top_neg_idx = scores.argsort()[:n]
    features = tfidf.get_feature_names_out()
    pos_words = [(features[i], scores[i]) for i in top_pos_idx if scores[i] > 0]
    neg_words = [(features[i], abs(scores[i])) for i in top_neg_idx if scores[i] < 0]
    return pos_words, neg_words

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Type a movie review below and the model will predict whether it's **positive** or **negative**.")
st.markdown("---")

# Example reviews
with st.expander("💡 Try an example"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Positive example"):
            st.session_state['review_text'] = (
                "This film is absolutely brilliant! The acting is superb, "
                "the story is captivating and the ending left me speechless. "
                "One of the best movies I have ever seen. Highly recommended!"
            )
    with col2:
        if st.button("Negative example"):
            st.session_state['review_text'] = (
                "What a terrible waste of time. The plot makes no sense, "
                "the acting is awful and the dialogue is painfully boring. "
                "I walked out after 30 minutes. Avoid at all costs."
            )

review_text = st.text_area(
    "Your review:",
    value=st.session_state.get('review_text', ''),
    height=150,
    placeholder="Write your movie review here..."
)

analyze = st.button("Analyze Sentiment", type="primary", use_container_width=True)

if analyze and review_text.strip():
    cleaned = clean(review_text)
    vec = tfidf.transform([cleaned])
    prob = model.predict_proba(vec)[0]
    pred = int(model.predict(vec)[0])

    st.markdown("---")

    # Result
    if pred == 1:
        st.success(f"### 😊 POSITIVE  —  {prob[1]*100:.1f}% confidence")
    else:
        st.error(f"### 😞 NEGATIVE  —  {prob[0]*100:.1f}% confidence")

    # Probability bar
    col1, col2 = st.columns(2)
    col1.metric("Positive probability", f"{prob[1]*100:.1f}%")
    col2.metric("Negative probability", f"{prob[0]*100:.1f}%")

    st.progress(float(prob[1]))

    # Influential words
    pos_words, neg_words = get_top_words(cleaned)
    st.markdown("---")
    st.markdown("#### Words that influenced the prediction")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Positive signals 🟢**")
        if pos_words:
            for word, score in pos_words:
                st.markdown(f"- `{word}` ({score:.3f})")
        else:
            st.markdown("*None detected*")
    with col2:
        st.markdown("**Negative signals 🔴**")
        if neg_words:
            for word, score in neg_words:
                st.markdown(f"- `{word}` ({score:.3f})")
        else:
            st.markdown("*None detected*")

elif analyze and not review_text.strip():
    st.warning("Please write a review first.")

st.markdown("---")
st.caption("Model: Logistic Regression + TF-IDF · Trained on NLTK movie_reviews corpus · Accuracy: 83.5%")
