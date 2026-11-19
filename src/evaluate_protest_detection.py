
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load model and evaluation dataset
model = joblib.load("models/event_detection_tfidf.joblib")
df = pd.read_csv("data/processed/protest_evaluation.csv")

X = df["tweet"].astype(str)
y_true = df["label"]

# Predict
y_pred = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

# Display individual predictions
df["predicted_label"] = y_pred
df["event_probability"] = probabilities.round(4)

print("\nINDIVIDUAL PREDICTIONS")
print("=" * 65)

for _, row in df.iterrows():
    actual = "Protest" if row["label"] == 1 else "Non-protest"
    predicted = "Event" if row["predicted_label"] == 1 else "Non-event"

    print(f"\nTweet: {row['tweet']}")
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")
    print(f"Event probability: {row['event_probability']}")

# Overall evaluation
print("\n\nPROTEST-SPECIFIC EVALUATION")
print("=" * 65)

print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("Precision:", round(precision_score(y_true, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_true, y_pred, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_true, y_pred, zero_division=0), 4))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Non-protest", "Protest"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save detailed predictions
df.to_csv(
    "results/protest_detection_evaluation.csv",
    index=False
)

print("\nDetailed results saved.")
