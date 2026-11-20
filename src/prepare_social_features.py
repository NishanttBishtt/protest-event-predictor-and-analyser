
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_PATH = "data/processed/combined_forecasting_data.csv"
OUTPUT_PATH = "data/processed/combined_forecasting_features.csv"

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["country", "date"]).reset_index(drop=True)

# Reconstruct daily score sum so rolling averages are
# weighted by the number of tweets on each day.
df["score_sum"] = (
    df["mean_unrest_score"] * df["tweet_count"]
)

# Features that will be calculated for each country.
feature_columns = []

windows = {
    7: 5,    # Require at least 5 observed days
    14: 10,  # Require at least 10 observed days
    28: 20   # Require at least 20 observed days
}

for window, min_days in windows.items():

    print(f"Creating {window}-day social features...")

    # Shift by one day to avoid using information
    # from the prediction date itself.
    for country, group in df.groupby("country"):

        idx = group.index

        tweets = group["tweet_count"].shift(1)
        unrest = group["unrest_tweet_count"].shift(1)
        scores = group["score_sum"].shift(1)
        max_scores = group["max_unrest_score"].shift(1)

        observed = group["gcut_available"].shift(1)

        tweet_roll = tweets.rolling(
            window, min_periods=1
        ).sum()

        unrest_roll = unrest.rolling(
            window, min_periods=1
        ).sum()

        score_roll = scores.rolling(
            window, min_periods=1
        ).sum()

        observed_days = observed.rolling(
            window, min_periods=1
        ).sum()

        max_roll = max_scores.rolling(
            window, min_periods=1
        ).max()

        # Average daily volume across observed days.
        avg_tweets = tweet_roll / observed_days

        # Proportion of unrest-related tweets.
        unrest_ratio = unrest_roll / tweet_roll

        # Tweet-weighted average unrest score.
        mean_score = score_roll / tweet_roll

        # Do not use windows with insufficient coverage.
        valid = observed_days >= min_days

        prefix = f"past_{window}d"

        df.loc[idx, f"{prefix}_avg_tweets"] = (
            avg_tweets.where(valid).values
        )

        df.loc[idx, f"{prefix}_unrest_ratio"] = (
            unrest_ratio.where(valid).values
        )

        df.loc[idx, f"{prefix}_mean_score"] = (
            mean_score.where(valid).values
        )

        df.loc[idx, f"{prefix}_max_score"] = (
            max_roll.where(valid).values
        )

        df.loc[idx, f"{prefix}_observed_days"] = (
            observed_days.values
        )

    feature_columns.extend([
        f"past_{window}d_avg_tweets",
        f"past_{window}d_unrest_ratio",
        f"past_{window}d_mean_score",
        f"past_{window}d_max_score",
        f"past_{window}d_observed_days"
    ])

# Save engineered dataset
Path("data/processed").mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\n========== SOCIAL FEATURE ENGINEERING ==========")
print("Shape:", df.shape)
print("Countries:", df["country"].nunique())
print("Date range:", df["date"].min(), "to", df["date"].max())

print("\nGenerated features:")
print(feature_columns)

print("\nMissing values in social features:")
print(df[feature_columns].isnull().sum())

print("\nSaved:", OUTPUT_PATH)
