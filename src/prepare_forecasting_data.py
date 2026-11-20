
import pandas as pd
from pathlib import Path

DATA_PATH = Path(
    "civil-unrest-case-study/data/2014-01-01-2020-01-01_acled_reduced_all.csv"
)
OUTPUT_PATH = Path("data/processed")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Load ACLED data
df = pd.read_csv(DATA_PATH)

# Keep protest events only, excluding riots
df["event_date"] = pd.to_datetime(
    df["event_date"], format="%d %B %Y"
)

df = df[
    (df["event_type"] == "Protests")
    & (df["event_date"] >= "2014-01-01")
    & (df["event_date"] < "2020-01-01")
].copy()

# Count protest events per country per day
daily = (
    df.groupby(["iso3", "event_date"])
    .size()
    .rename("event_count")
    .reset_index()
)

# Create a complete daily calendar for each country
dates = pd.date_range("2014-01-01", "2019-12-31", freq="D")
countries = sorted(daily["iso3"].unique())

calendar = pd.MultiIndex.from_product(
    [countries, dates],
    names=["iso3", "event_date"]
).to_frame(index=False)

daily = calendar.merge(
    daily, on=["iso3", "event_date"], how="left"
)

daily["event_count"] = daily["event_count"].fillna(0).astype(int)
daily = daily.sort_values(["iso3", "event_date"])

# Create features from the past only
grouped = daily.groupby("iso3")["event_count"]

for window in [7, 14, 28]:
    daily[f"past_{window}d_events"] = grouped.transform(
        lambda s: s.shift(1).rolling(window, min_periods=window).sum()
    )

# Target: any protest in the NEXT 7 days, excluding today
daily["future_7d_events"] = grouped.transform(
    lambda s: s.shift(-1).rolling(7, min_periods=7).sum()
).groupby(daily["iso3"]).shift(-6)

# The target expression above is easier to validate with a direct
# country-wise calculation.
daily["future_7d_events"] = (
    daily.groupby("iso3")["event_count"]
    .transform(
        lambda s: pd.concat(
            [s.shift(-i) for i in range(1, 8)], axis=1
        ).sum(axis=1, min_count=7)
    )
)

daily["future_protest"] = (
    daily["future_7d_events"] > 0
).astype("float")

# Remove rows without enough past or future data
daily = daily.dropna(
    subset=["past_7d_events", "past_14d_events",
            "past_28d_events", "future_7d_events"]
)

# Save prepared data
daily.to_csv(OUTPUT_PATH / "daily_protest_forecasting.csv", index=False)

print("Prepared dataset shape:", daily.shape)
print("Countries:", daily["iso3"].nunique())
print("Date range:", daily["event_date"].min(), "to", daily["event_date"].max())
print("\nFuture protest target distribution:")
print(daily["future_protest"].value_counts(normalize=True).rename("proportion"))
print("\nSample:")
print(daily.head())
print("\nSaved to:", OUTPUT_PATH / "daily_protest_forecasting.csv")
