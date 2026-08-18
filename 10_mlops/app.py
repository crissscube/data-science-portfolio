"""
Step 2 of the MLOps pipeline: serve the trained model as a REST API.

The API loads model.pkl ONCE at startup, then answers prediction requests.
This is how a model goes from "a file on disk" to "a service other systems
can call over HTTP".
"""

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model artifact once, when the API starts up.
model = joblib.load("model.pkl")

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts whether a piece of text is positive or negative.",
    version="1.0.0",
)


# Defines the expected request body: a JSON object with a "text" field.
class Review(BaseModel):
    text: str


@app.get("/")
def health_check():
    """Simple endpoint to confirm the API is up."""
    return {"status": "ok", "message": "Sentiment Analysis API is running"}


@app.post("/predict")
def predict(review: Review):
    """Take a text and return its predicted sentiment with a confidence score."""
    probabilities = model.predict_proba([review.text])[0]
    classes = model.classes_
    best_index = probabilities.argmax()

    return {
        "text": review.text,
        "sentiment": classes[best_index],
        "confidence": round(float(probabilities[best_index]), 4),
    }
