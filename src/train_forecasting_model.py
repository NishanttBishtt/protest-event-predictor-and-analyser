
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = Path("data/processed/daily_protest_forecasting.csv")
MODEL_PATH = Path("models/protest_forecasting_model.joblib")
RESULTS_PATH = Path("results/forecasting_metrics.txt")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["event_date"])

features = ["past_7d_events", "past_14d_events", "past_28d_events"]
target = "future_protest"

train = df[df["event_date"] <= "2017-12-24"]

valid = df[
    (df["event_date"] >= "2018-01-01")
    & (df["event_date"] <= "2018-12-24")
]

test = df[
    (df["event_date"] >= "2019-01-01")
    & (df["event_date"] <= "2019-12-24")
]

X_train, y_train = train[features], train[target].astype(int)
X_valid, y_valid = valid[features], valid[target].astype(int)
X_test, y_test = test[features], test[target].astype(int)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

def evaluate(name, y_true, predictions):
    return {
        "dataset": name,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0)
    }

# Select the classification threshold using validation data only.
valid_probabilities = model.predict_proba(X_valid)[:, 1]
thresholds = np.arange(0.20, 0.81, 0.05)

best_threshold = max(
    thresholds,
    key=lambda t: f1_score(
        y_valid, (valid_probabilities >= t).astype(int),
        zero_division=0
    )
)

test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

baseline_predictions = np.ones(len(y_test), dtype=int)

results = [
    evaluate("Always predict protest baseline", y_test, baseline_predictions),
    evaluate("Random Forest", y_test, test_predictions)
]

results_df = pd.DataFrame(results)
print("Training rows:", len(train))
print("Validation rows:", len(valid))
print("Test rows:", len(test))
print("Selected threshold:", round(best_threshold, 2))
print("\nTest results:")
print(results_df.to_string(index=False))

print("\nRandom Forest classification report:")
print(classification_report(y_test, test_predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions))

results_df.to_csv("results/forecasting_metrics.csv", index=False)

with open(RESULTS_PATH, "w") as f:
    f.write(f"Selected threshold: {best_threshold:.2f}\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")
    f.write(classification_report(y_test, test_predictions, zero_division=0))
    f.write("\nConfusion matrix:\n")
    f.write(str(confusion_matrix(y_test, test_predictions)))

joblib.dump(
    {"model": model, "features": features, "threshold": best_threshold},
    MODEL_PATH
)

print("\nSaved model to:", MODEL_PATH)
print("Saved metrics to:", RESULTS_PATH)
