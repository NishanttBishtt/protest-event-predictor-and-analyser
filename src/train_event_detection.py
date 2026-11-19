import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# Load dataset
df = pd.read_csv("data/processed/dipd_detection.csv")

X = df["tweet"].astype(str)
y = df["label"]

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# TF-IDF + Logistic Regression pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("Training event detection model...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("F1 Score:", round(f1_score(y_test, predictions), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Non-event", "Event"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/event_detection_tfidf.joblib")

print("\nModel saved to models/event_detection_tfidf.joblib")
