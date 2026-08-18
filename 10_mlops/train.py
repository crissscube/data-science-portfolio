"""
Step 1 of the MLOps pipeline: train a sentiment model and save it as a
reusable artifact (model.pkl) that the API will later load and serve.

Key MLOps idea: TRAINING and SERVING are separate steps. We train once,
save the model to disk, and the API just loads that saved artifact.

The dataset is kept small and self-contained on purpose. The focus of this
project is the MLOps workflow (train -> package -> serve -> containerize),
not squeezing out maximum accuracy.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# A small, self-contained labeled dataset (movie-review style).
POSITIVE = [
    "this movie was amazing and truly inspiring",
    "an absolute masterpiece, I loved every minute",
    "brilliant acting and a beautiful story",
    "fantastic film, highly recommended",
    "a wonderful and heartwarming experience",
    "the best movie I have seen this year",
    "incredible visuals and a gripping plot",
    "superb direction and outstanding performances",
    "a delightful and charming comedy",
    "emotional, powerful and unforgettable",
    "excellent script with great characters",
    "a thrilling and exciting adventure",
    "genuinely funny and very entertaining",
    "stunning cinematography and a moving soundtrack",
    "a perfect blend of drama and humor",
    "captivating from start to finish",
    "smart, original and deeply satisfying",
    "loved it, would watch it again",
    "a beautiful film that touched my heart",
    "impressive and thoroughly enjoyable",
]

NEGATIVE = [
    "this movie was terrible and boring",
    "a complete waste of time and money",
    "awful acting and a weak plot",
    "the worst film I have ever seen",
    "dull, slow and painfully predictable",
    "poorly written with flat characters",
    "a disappointing and forgettable mess",
    "badly directed and hard to watch",
    "cringe worthy dialogue and bad pacing",
    "an absolute disaster from start to finish",
    "confusing, dragging and pointless",
    "cheap effects and a lazy story",
    "I hated every minute of it",
    "unoriginal, generic and uninspired",
    "a boring film with no real ending",
    "the acting was wooden and unconvincing",
    "not funny, not clever, just annoying",
    "tedious and far too long",
    "a dull and lifeless experience",
    "avoid this film, it is truly bad",
]

texts = POSITIVE + NEGATIVE
labels = ["pos"] * len(POSITIVE) + ["neg"] * len(NEGATIVE)

# A Pipeline bundles the text vectorizer + classifier into ONE object.
# This matters: when we save it, the API can take raw text and predict
# directly, without redoing the preprocessing separately.
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000)),
])

model.fit(texts, labels)

# Quick sanity check on a couple of unseen phrases.
samples = ["what a fantastic and moving film", "this was a boring waste of time"]
for text, pred in zip(samples, model.predict(samples)):
    print(f"  '{text}' -> {pred}")

# Save the trained pipeline as a single artifact.
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")
