
import pandas as pd

PATH = "data/processed/combined_forecasting_features.csv"

df = pd.read_csv(PATH)
df["date"] = pd.to_datetime(df["date"])

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

# Same chronological split as the baseline experiment
splits = {
    "Train": df[df["date"] <= "2017-12-24"],
    "Validation": df[
        (df["date"] >= "2018-01-01") &
        (df["date"] <= "2018-12-24")
    ],
    "Test": df[
        (df["date"] >= "2019-01-01") &
        (df["date"] <= "2019-12-24")
    ]
}

print("========== MODEL B DATA AVAILABILITY ==========")

for name, part in splits.items():

    complete = part[social_features].notna().all(axis=1)

    print(f"\n{name}")
    print("Total observations:", len(part))
    print("G-CUT matched observations:",
          part["gcut_available"].sum())
    print("Complete social-feature observations:",
          complete.sum())
    print("Incomplete social-feature observations:",
          (~complete).sum())

    usable = part.loc[complete]

    print("Positive targets:",
          (usable["future_protest"] == 1).sum())
    print("Negative targets:",
          (usable["future_protest"] == 0).sum())

print("\n========== FEATURE MISSINGNESS ==========")
print(df[social_features].isna().sum())
