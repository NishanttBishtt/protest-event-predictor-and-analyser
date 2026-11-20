
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
    confusion_matrix,
    classification_report
)

DATA_PATH = "data/processed/combined_forecasting_features.csv"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# Historical features
historical_features = [
    "past_7d_events",
    "past_14d_events",
    "past_28d_events"
]

# Social-media features
social_features = [
    f"past_{window}d_{feature}"
    for window in [7, 14, 28]
    for feature in [
        "avg_tweets",
        "unrest_ratio",
        "mean_score",
        "max_score",
        "observed_days"
    ]
]

# Keep observations with all social features available.
# Both models will use this exact same subset.
df = df.dropna(
    subset=historical_features + social_features + ["future_protest"]
).copy()

df["future_protest"] = df["future_protest"].astype(int)

# Chronological split
train = df[df["date"] <= "2017-12-24"]
valid = df[
    (df["date"] >= "2018-01-01") &
    (df["date"] <= "2018-12-24")
]
test = df[
    (df["date"] >= "2019-01-01") &
    (df["date"] <= "2019-12-24")
]

print("========== MATCHED DATASET ==========")
print("Train:", len(train))
print("Validation:", len(valid))
print("Test:", len(test))

assert len(train) > 0 and len(valid) > 0 and len(test) > 0

# Verify that both models use identical observations
print("\nPositive test observations:",
      test["future_protest"].sum())
print("Negative test observations:",
      (test["future_protest"] == 0).sum())


def build_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


def select_threshold(y_true, probabilities):
    thresholds = np.arange(0.20, 0.81, 0.05)

    scores = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        scores.append(
            f1_score(y_true, predictions, zero_division=0)
        )

    best_index = np.argmax(scores)

    return thresholds[best_index], scores[best_index]


def evaluate_model(name, features):

    model = build_model()

    X_train = train[features]
    y_train = train["future_protest"]

    X_valid = valid[features]
    y_valid = valid["future_protest"]

    X_test = test[features]
    y_test = test["future_protest"]

    print(f"\n========== {name} ==========")

    model.fit(X_train, y_train)

    # Select threshold using validation data only
    valid_probabilities = model.predict_proba(X_valid)[:, 1]

    threshold, valid_f1 = select_threshold(
        y_valid, valid_probabilities
    )

    # Evaluate on untouched test data
    test_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (test_probabilities >= threshold).astype(int)

    metrics = {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    print("Selected threshold:", threshold)
    print("Validation F1:", round(valid_f1, 4))
    print("\nTest metrics:")

    for key, value in metrics.items():
        if key != "Model":
            print(f"{key}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    return model, metrics


# Train both models on identical observations
model_a, metrics_a = evaluate_model(
    "Model A - Historical Only",
    historical_features
)

model_b, metrics_b = evaluate_model(
    "Model B - Historical + Social Media",
    historical_features + social_features
)

# Comparison
comparison = pd.DataFrame([metrics_a, metrics_b])

print("\n========== FINAL COMPARISON ==========")
print(comparison.to_string(index=False))

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

joblib.dump(
    model_a,
    "models/protest_model_historical.joblib"
)

joblib.dump(
    model_b,
    "models/protest_model_combined.joblib"
)

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModels and comparison saved successfully.")
