
import pandas as pd
from pathlib import Path

ACLED_PATH = "data/processed/daily_protest_forecasting.csv"
GCUT_PATH = "data/processed/gcut_daily_features_2014_2019.csv"

OUTPUT_PATH = "data/processed/combined_forecasting_data.csv"
COVERAGE_PATH = "results/combined_dataset_coverage.csv"

# Load datasets
acled = pd.read_csv(ACLED_PATH)
gcut = pd.read_csv(GCUT_PATH)

# Standardize join columns
acled = acled.rename(columns={
    "iso3": "country",
    "event_date": "date"
})

gcut = gcut.rename(columns={
    "country_code_alpha3": "country"
})

acled["date"] = pd.to_datetime(acled["date"])
gcut["date"] = pd.to_datetime(gcut["date"])

acled["country"] = acled["country"].str.strip()
gcut["country"] = gcut["country"].str.strip()

# Identify common countries
common_countries = (
    set(acled["country"].unique())
    & set(gcut["country"].unique())
)

print("Common countries:", len(common_countries))
print("Excluded ACLED-only countries:",
      sorted(set(acled["country"]) - common_countries))

# Restrict both datasets to common countries
acled = acled[
    acled["country"].isin(common_countries)
].copy()

gcut = gcut[
    gcut["country"].isin(common_countries)
].copy()

# Check uniqueness
assert not acled.duplicated(["country", "date"]).any()
assert not gcut.duplicated(["country", "date"]).any()

# Left join: preserve every eligible ACLED observation
combined = acled.merge(
    gcut,
    on=["country", "date"],
    how="left",
    validate="one_to_one",
    indicator=True
)

# Explicit data availability indicator
combined["gcut_available"] = (
    combined["_merge"] == "both"
).astype(int)

combined = combined.drop(columns=["_merge"])

# Sort data
combined = combined.sort_values(
    ["country", "date"]
).reset_index(drop=True)

# Coverage summary
coverage = (
    combined.groupby("country")
    .agg(
        total_acled_days=("date", "size"),
        available_gcut_days=("gcut_available", "sum")
    )
)

coverage["coverage_percentage"] = (
    coverage["available_gcut_days"]
    / coverage["total_acled_days"] * 100
)

# Save outputs
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(parents=True, exist_ok=True)

combined.to_csv(OUTPUT_PATH, index=False)
coverage.to_csv(COVERAGE_PATH)

print("\n========== INTEGRATION RESULTS ==========")
print("Combined shape:", combined.shape)
print("Countries:", combined["country"].nunique())
print("Matched rows:", combined["gcut_available"].sum())
print("Unmatched rows:", (combined["gcut_available"] == 0).sum())
print(
    "Overall coverage:",
    round(combined["gcut_available"].mean() * 100, 2),
    "%"
)
print("Missing G-CUT values:")
print(combined[
    ["tweet_count", "unrest_tweet_count",
     "mean_unrest_score", "max_unrest_score"]
].isnull().sum())

print("\nSaved:", OUTPUT_PATH)
print("Saved:", COVERAGE_PATH)
